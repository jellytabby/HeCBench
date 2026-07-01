#ifndef GMM_COMMON_CPU_REFERENCE_H
#define GMM_COMMON_CPU_REFERENCE_H

void invert_cpu_natural_log(float* data, int actualsize, float* log_determinant) {
  int maxsize = actualsize;
  int n = actualsize;
  *log_determinant = 0.0f;

  if (actualsize == 1) {
    *log_determinant = ::logf(data[0]);
    data[0] = 1.0f / data[0];
  } else if(actualsize >= 2) {
    for (int i=1; i < actualsize; i++) data[i] /= data[0];
    for (int i=1; i < actualsize; i++) {
      for (int j=i; j < actualsize; j++) {
        float sum = 0.0f;
        for (int k = 0; k < i; k++)
          sum += data[j*maxsize+k] * data[k*maxsize+i];
        data[j*maxsize+i] -= sum;
      }
      if (i == actualsize-1) continue;
      for (int j=i+1; j < actualsize; j++) {
        float sum = 0.0f;
        for (int k = 0; k < i; k++)
          sum += data[i*maxsize+k]*data[k*maxsize+j];
        data[i*maxsize+j] =
          (data[i*maxsize+j]-sum) / data[i*maxsize+i];
      }
    }

    for(int i=0; i<actualsize; i++) {
      *log_determinant += ::logf(fabs(data[i*n+i]));
    }
    for (int i = 0; i < actualsize; i++)
      for (int j = i; j < actualsize; j++) {
        float x = 1.0f;
        if (i != j) {
          x = 0.0f;
          for (int k = i; k < j; k++)
            x -= data[j*maxsize+k]*data[k*maxsize+i];
        }
        data[j*maxsize+i] = x / data[j*maxsize+j];
      }
    for (int i = 0; i < actualsize; i++)
      for (int j = i; j < actualsize; j++) {
        if (i == j) continue;
        float sum = 0.0f;
        for (int k = i; k < j; k++)
          sum += data[k*maxsize+j]*((i==k) ? 1.0f : data[i*maxsize+k]);
        data[i*maxsize+j] = -sum;
      }
    for (int i = 0; i < actualsize; i++)
      for (int j = 0; j < actualsize; j++) {
        float sum = 0.0f;
        for (int k = ((i>j)?i:j); k < actualsize; k++)
          sum += ((j==k)?1.0f:data[j*maxsize+k])*data[k*maxsize+i];
        data[j*maxsize+i] = sum;
      }
  } else {
    PRINT("Error: Invalid dimensionality for invert_cpu_natural_log(...)\n");
  }
}

void compute_constants_cpu(clusters_t* clusters, int num_clusters, int num_dimensions) {
  float sum = 0.0f;
  for(int c=0; c<num_clusters; c++) {
    sum += clusters->N[c];
  }

  for(int c=0; c<num_clusters; c++) {
    float log_determinant;
    memcpy(&clusters->Rinv[c*num_dimensions*num_dimensions],
        &clusters->R[c*num_dimensions*num_dimensions],
        sizeof(float)*num_dimensions*num_dimensions);
#if DIAG_ONLY
    log_determinant = 0.0f;
    for(int d=0; d<num_dimensions; d++) {
      float diag = clusters->Rinv[c*num_dimensions*num_dimensions+d*num_dimensions+d];
      log_determinant += ::logf(diag);
      clusters->Rinv[c*num_dimensions*num_dimensions+d*num_dimensions+d] = 1.0f / diag;
    }
#else
    invert_cpu_natural_log(&clusters->Rinv[c*num_dimensions*num_dimensions],
        num_dimensions, &log_determinant);
#endif
    clusters->constant[c] = -num_dimensions*0.5f*::logf(2.0f*PI) - 0.5f*log_determinant;
    clusters->pi[c] = (clusters->N[c] < 0.5f) ? 1e-10f : clusters->N[c] / sum;
  }
}

void seed_clusters_cpu(const float* fcs_data_by_event, clusters_t* clusters,
    int num_dimensions, int num_clusters, int num_events) {
  float means[NUM_DIMENSIONS];
  float variances[NUM_DIMENSIONS];
  float total_variance = 0.0f;

  for(int d=0; d<num_dimensions; d++) {
    means[d] = 0.0f;
    for(int e=0; e<num_events; e++) {
      means[d] += fcs_data_by_event[e*num_dimensions+d];
    }
    means[d] /= (float) num_events;
  }

  for(int d=0; d<num_dimensions; d++) {
    variances[d] = 0.0f;
    for(int e=0; e<num_events; e++) {
      float value = fcs_data_by_event[e*num_dimensions+d];
      variances[d] += value * value;
    }
    variances[d] /= (float) num_events;
    variances[d] -= means[d] * means[d];
    total_variance += variances[d];
  }

  float avgvar = total_variance / (float) num_dimensions / COVARIANCE_DYNAMIC_RANGE;
  float seed = (num_clusters > 1) ? (num_events-1.0f)/(num_clusters-1.0f) : 0.0f;

  for(int c=0; c<num_clusters; c++) {
    for(int d=0; d<num_dimensions; d++) {
      clusters->means[c*num_dimensions+d] =
        fcs_data_by_event[((int)(c*seed))*num_dimensions+d];
    }
    for(int row=0; row<num_dimensions; row++) {
      for(int col=0; col<num_dimensions; col++) {
        clusters->R[c*num_dimensions*num_dimensions+row*num_dimensions+col] =
          (row == col) ? 1.0f : 0.0f;
      }
    }
    clusters->pi[c] = 1.0f/((float)num_clusters);
    clusters->N[c] = ((float) num_events) / ((float)num_clusters);
    clusters->avgvar[c] = avgvar;
  }
}

float estep_cpu(const float* fcs_data_by_dimension, clusters_t* clusters,
    int num_dimensions, int num_clusters, int num_events) {
  float likelihood = 0.0f;

  for(int c=0; c<num_clusters; c++) {
    const float* means = &clusters->means[c*num_dimensions];
    const float* Rinv = &clusters->Rinv[c*num_dimensions*num_dimensions];
    float cluster_pi = clusters->pi[c];
    float constant = clusters->constant[c];

    for(int event=0; event<num_events; event++) {
      float like = 0.0f;
#if DIAG_ONLY
      for(int i=0; i<num_dimensions; i++) {
        float diff = fcs_data_by_dimension[i*num_events+event]-means[i];
        like += diff * diff * Rinv[i*num_dimensions+i];
      }
#else
      for(int i=0; i<num_dimensions; i++) {
        float diff_i = fcs_data_by_dimension[i*num_events+event]-means[i];
        for(int j=0; j<num_dimensions; j++) {
          float diff_j = fcs_data_by_dimension[j*num_events+event]-means[j];
          like += diff_i * diff_j * Rinv[i*num_dimensions+j];
        }
      }
#endif
      clusters->memberships[c*num_events+event] = -0.5f * like + constant + ::logf(cluster_pi);
    }
  }

  for(int event=0; event<num_events; event++) {
    float max_likelihood = clusters->memberships[event];
    for(int c=1; c<num_clusters; c++) {
      max_likelihood = fmaxf(max_likelihood,clusters->memberships[c*num_events+event]);
    }

    float denominator_sum = 0.0f;
    for(int c=0; c<num_clusters; c++) {
      denominator_sum += ::expf(clusters->memberships[c*num_events+event]-max_likelihood);
    }
    denominator_sum = max_likelihood + ::logf(denominator_sum);
    likelihood += denominator_sum;

    for(int c=0; c<num_clusters; c++) {
      clusters->memberships[c*num_events+event] =
        ::expf(clusters->memberships[c*num_events+event] - denominator_sum);
    }
  }

  return likelihood;
}

void mstep_cpu(const float* fcs_data_by_dimension, clusters_t* clusters,
    int num_dimensions, int num_clusters, int num_events) {
  for(int c=0; c<num_clusters; c++) {
    float sum = 0.0f;
    for(int event=0; event<num_events; event++) {
      sum += clusters->memberships[c*num_events+event];
    }
    clusters->N[c] = sum;
    clusters->pi[c] = sum;
  }

  for(int c=0; c<num_clusters; c++) {
    for(int d=0; d<num_dimensions; d++) {
      float sum = 0.0f;
      for(int event=0; event<num_events; event++) {
        sum += fcs_data_by_dimension[d*num_events+event] *
          clusters->memberships[c*num_events+event];
      }
      clusters->means[c*num_dimensions+d] =
        (clusters->N[c] > 0.5f) ? sum / clusters->N[c] : 0.0f;
    }
  }

  for(int c=0; c<num_clusters; c++) {
    if(clusters->N[c] > 0.5f) {
      for(int row=0; row<num_dimensions; row++) {
        for(int col=0; col<=row; col++) {
#if DIAG_ONLY
          if(row != col) {
            clusters->R[c*num_dimensions*num_dimensions+row*num_dimensions+col] = 0.0f;
            clusters->R[c*num_dimensions*num_dimensions+col*num_dimensions+row] = 0.0f;
            continue;
          }
#endif
          float cov_sum = 0.0f;
          for(int event=0; event<num_events; event++) {
            cov_sum += (fcs_data_by_dimension[row*num_events+event] -
                clusters->means[c*num_dimensions+row]) *
              (fcs_data_by_dimension[col*num_events+event] -
                clusters->means[c*num_dimensions+col]) *
              clusters->memberships[c*num_events+event];
          }
          if(row == col) {
            cov_sum += clusters->avgvar[c];
          }
          cov_sum /= clusters->N[c];
          clusters->R[c*num_dimensions*num_dimensions+row*num_dimensions+col] = cov_sum;
          clusters->R[c*num_dimensions*num_dimensions+col*num_dimensions+row] = cov_sum;
        }
      }
    } else {
      for(int row=0; row<num_dimensions; row++) {
        for(int col=0; col<num_dimensions; col++) {
          clusters->R[c*num_dimensions*num_dimensions+row*num_dimensions+col] =
            (row == col) ? 1.0f : 0.0f;
        }
      }
    }
  }
}

void copy_active_clusters(clusters_t* dest, const clusters_t* src,
    int num_clusters, int num_dimensions, int num_events) {
  memcpy(dest->N,src->N,sizeof(float)*num_clusters);
  memcpy(dest->pi,src->pi,sizeof(float)*num_clusters);
  memcpy(dest->constant,src->constant,sizeof(float)*num_clusters);
  memcpy(dest->avgvar,src->avgvar,sizeof(float)*num_clusters);
  memcpy(dest->means,src->means,sizeof(float)*num_dimensions*num_clusters);
  memcpy(dest->R,src->R,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
  memcpy(dest->Rinv,src->Rinv,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
  memcpy(dest->memberships,src->memberships,sizeof(float)*num_events*num_clusters);
}

clusters_t* cluster_cpu_reference(int original_num_clusters, int desired_num_clusters,
    int* final_num_clusters, int num_dimensions, int num_events,
    float* fcs_data_by_event) {
  int ideal_num_clusters = original_num_clusters;
  int stop_number = (desired_num_clusters == 0) ? 1 : desired_num_clusters;

  float* fcs_data_by_dimension = (float*) malloc(sizeof(float)*num_events*num_dimensions);
  for(int e=0; e<num_events; e++) {
    for(int d=0; d<num_dimensions; d++) {
      if(isnan(fcs_data_by_event[e*num_dimensions+d])) {
        printf("Error: Found NaN value in input data. Exiting.\n");
        free(fcs_data_by_dimension);
        return NULL;
      }
      fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
    }
  }

  clusters_t clusters;
  setupCluster(&clusters, original_num_clusters, num_events, num_dimensions);

  clusters_t *saved_clusters = (clusters_t*) malloc(sizeof(clusters_t));
  setupCluster(saved_clusters, original_num_clusters, num_events, num_dimensions);

  clusters_t scratch_cluster;
  setupCluster(&scratch_cluster, 1, num_events, num_dimensions);

  seed_clusters_cpu(fcs_data_by_event, &clusters, num_dimensions, original_num_clusters, num_events);
  compute_constants_cpu(&clusters, original_num_clusters, num_dimensions);

  float epsilon = (1+num_dimensions+0.5f*(num_dimensions+1)*num_dimensions)*
    ::logf((float)num_events*num_dimensions)*0.001f;
  float min_rissanen = FLT_MAX;

  for(int num_clusters=original_num_clusters; num_clusters >= stop_number; num_clusters--) {
    float likelihood = estep_cpu(fcs_data_by_dimension, &clusters,
        num_dimensions, num_clusters, num_events);
    float change = epsilon*2.0f;
    int iters = 0;

    while(iters < MIN_ITERS || (fabs(change) > epsilon && iters < MAX_ITERS)) {
      float old_likelihood = likelihood;
      mstep_cpu(fcs_data_by_dimension, &clusters, num_dimensions, num_clusters, num_events);
      compute_constants_cpu(&clusters, num_clusters, num_dimensions);
      likelihood = estep_cpu(fcs_data_by_dimension, &clusters,
          num_dimensions, num_clusters, num_events);
      change = likelihood - old_likelihood;
      iters++;
    }

    float rissanen = -likelihood + 0.5f*(num_clusters*(1.0f+num_dimensions+
          0.5f*(num_dimensions+1.0f)*num_dimensions)-1.0f)*
      ::logf((float)num_events*num_dimensions);

    if(num_clusters == original_num_clusters ||
        (rissanen < min_rissanen && desired_num_clusters == 0) ||
        (num_clusters == desired_num_clusters)) {
      min_rissanen = rissanen;
      ideal_num_clusters = num_clusters;
      copy_active_clusters(saved_clusters, &clusters,
          num_clusters, num_dimensions, num_events);
    }

    if(num_clusters > stop_number) {
      for(int i=num_clusters-1; i >= 0; i--) {
        if(clusters.N[i] < 0.5f) {
          for(int j=i; j < num_clusters-1; j++) {
            copy_cluster(clusters,j,clusters,j+1,num_dimensions);
          }
          num_clusters--;
        }
      }

      int min_c1 = 0;
      int min_c2 = 1;
      float min_distance = 0.0f;
      for(int c1=0; c1<num_clusters;c1++) {
        for(int c2=c1+1; c2<num_clusters;c2++) {
          float distance = cluster_distance(clusters,c1,c2,scratch_cluster,num_dimensions);
          if((c1 == 0 && c2 == 1) || distance < min_distance) {
            min_distance = distance;
            min_c1 = c1;
            min_c2 = c2;
          }
        }
      }

      add_clusters(clusters,min_c1,min_c2,scratch_cluster,num_dimensions);
      copy_cluster(clusters,min_c1,scratch_cluster,0,num_dimensions);

      for(int i=min_c2; i < num_clusters-1; i++) {
        copy_cluster(clusters,i,clusters,i+1,num_dimensions);
      }
    }
  }

  freeCluster(&scratch_cluster);
  freeCluster(&clusters);
  free(fcs_data_by_dimension);

  *final_num_clusters = ideal_num_clusters;
  return saved_clusters;
}

#endif

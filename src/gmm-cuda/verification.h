#ifndef GMM_COMMON_VERIFICATION_H
#define GMM_COMMON_VERIFICATION_H

std::vector<int> matchClustersByMeans(const clusters_t* gpu_clusters,
    const clusters_t* cpu_clusters, int num_clusters, int num_dimensions) {
  std::vector<int> mapping(num_clusters, -1);
  std::vector<bool> used(num_clusters, false);

  for(int gpu_c=0; gpu_c<num_clusters; gpu_c++) {
    float best_distance = FLT_MAX;
    int best_cpu_c = -1;

    for(int cpu_c=0; cpu_c<num_clusters; cpu_c++) {
      if(used[cpu_c]) {
        continue;
      }

      float distance = 0.0f;
      for(int d=0; d<num_dimensions; d++) {
        float diff = gpu_clusters->means[gpu_c*num_dimensions+d] -
          cpu_clusters->means[cpu_c*num_dimensions+d];
        distance += diff * diff;
      }

      if(best_cpu_c < 0 || distance < best_distance) {
        best_distance = distance;
        best_cpu_c = cpu_c;
      }
    }

    mapping[gpu_c] = best_cpu_c;
    if(best_cpu_c >= 0) {
      used[best_cpu_c] = true;
    }
  }

  return mapping;
}

bool compareMappedClusterArray(const char* name, const float* gpu, const float* cpu,
    const std::vector<int>& mapping, int num_clusters, int values_per_cluster,
    float abs_tol, float rel_tol) {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  int max_gpu_cluster = 0;
  int max_value = 0;

  for(int gpu_c=0; gpu_c<num_clusters; gpu_c++) {
    int cpu_c = mapping[gpu_c];
    for(int i=0; i<values_per_cluster; i++) {
      float gpu_value = gpu[gpu_c*values_per_cluster+i];
      float cpu_value = cpu[cpu_c*values_per_cluster+i];
      float abs_err = fabsf(gpu_value - cpu_value);
      float denom = fmaxf(fabsf(cpu_value), 1.0f);
      float rel_err = abs_err / denom;
      if(abs_err > max_abs) {
        max_abs = abs_err;
        max_rel = rel_err;
        max_gpu_cluster = gpu_c;
        max_value = i;
      }
    }
  }

  bool ok = max_abs <= abs_tol || max_rel <= rel_tol;
  printf("VERIFY %s: max_abs=%e max_rel=%e gpu_cluster=%d cpu_cluster=%d value=%d %s\n",
      name, max_abs, max_rel, max_gpu_cluster, mapping[max_gpu_cluster],
      max_value, ok ? "PASS" : "FAIL");
  return ok;
}

bool verifyWithCpuReference(clusters_t* gpu_clusters, int gpu_num_clusters,
    int original_num_clusters, int desired_num_clusters, int num_dimensions,
    int num_events, float* fcs_data_by_event) {
  int cpu_num_clusters = 0;
  clusters_t* cpu_clusters = cluster_cpu_reference(original_num_clusters,
      desired_num_clusters, &cpu_num_clusters, num_dimensions, num_events,
      fcs_data_by_event);

  if(!cpu_clusters) {
    printf("CPU reference failed to produce clusters.\n");
    return false;
  }

  bool ok = true;
  if(cpu_num_clusters != gpu_num_clusters) {
    printf("VERIFY final_num_clusters: GPU=%d CPU=%d FAIL\n",
        gpu_num_clusters, cpu_num_clusters);
    ok = false;
  } else {
    printf("VERIFY final_num_clusters: %d PASS\n", gpu_num_clusters);
  }

  int num_clusters = gpu_num_clusters < cpu_num_clusters ? gpu_num_clusters : cpu_num_clusters;
  std::vector<int> mapping = matchClustersByMeans(gpu_clusters, cpu_clusters,
      num_clusters, num_dimensions);

  ok &= compareMappedClusterArray("N", gpu_clusters->N, cpu_clusters->N,
      mapping, num_clusters, 1, 5.0f, 5e-2f);
  ok &= compareMappedClusterArray("pi", gpu_clusters->pi, cpu_clusters->pi,
      mapping, num_clusters, 1, 5e-3f, 5e-2f);
  ok &= compareMappedClusterArray("constant", gpu_clusters->constant, cpu_clusters->constant,
      mapping, num_clusters, 1, 2e-2f, 5e-2f);
  ok &= compareMappedClusterArray("avgvar", gpu_clusters->avgvar, cpu_clusters->avgvar,
      mapping, num_clusters, 1, 1e-6f, 1e-5f);
  ok &= compareMappedClusterArray("means", gpu_clusters->means, cpu_clusters->means,
      mapping, num_clusters, num_dimensions, 5e-2f, 5e-2f);
  ok &= compareMappedClusterArray("R", gpu_clusters->R, cpu_clusters->R,
      mapping, num_clusters, num_dimensions*num_dimensions, 5e-2f, 5e-2f);
  ok &= compareMappedClusterArray("Rinv", gpu_clusters->Rinv, cpu_clusters->Rinv,
      mapping, num_clusters, num_dimensions*num_dimensions, 5e-2f, 5e-2f);
  ok &= compareMappedClusterArray("memberships", gpu_clusters->memberships,
      cpu_clusters->memberships, mapping, num_clusters, num_events, 5e-2f, 5e-2f);

  freeCluster(cpu_clusters);
  free(cpu_clusters);

  printf("CPU reference verification: %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

#endif

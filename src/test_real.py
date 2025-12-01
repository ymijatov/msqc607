from real_data import describe_gesl_file
from real_data_experiments import quick_analysis, compare_with_known_frequencies
from visualization import quick_benchmark


### 1032 ###
print(describe_gesl_file(
    './../data/sigId-1032.csv',
    './../data/sigId-1032-Metadata.csv'
))


results_1032 = quick_analysis(
    './../data/sigId-1032.csv',
    './../data/sigId-1032-Metadata.csv',
    pmu_id='P001',
    segment_duration=15.0
)

compare_with_known_frequencies(results_1032)


quick_benchmark(
    './../data/sigId-1032.csv',
    './../data/sigId-1032-Metadata.csv',
    results_1032,
    save_path='./../figures/sigId-1032-benchmark.png'
)

### 989 ###

print(describe_gesl_file(
    './../data/sigId-989.csv',
    './../data/sigId-989-Metadata.csv'
))

results_989 = quick_analysis(
    './../data/sigId-989.csv',
    './../data/sigId-989-Metadata.csv',
    pmu_id='P001',
    segment_duration=15.0
)

compare_with_known_frequencies(results_989)

quick_benchmark(
    './../data/sigId-989.csv',
    './../data/sigId-989-Metadata.csv', 
    results_989,
    save_path='./../figures/sigId-989-benchmark.png'
)

### 1015 ###


print(describe_gesl_file(
    './../data/sigId-1015.csv',
    './../data/sigId-1015-Metadata.csv'
))

results_1015 = quick_analysis(
    './../data/sigId-1015.csv',
    './../data/sigId-1015-Metadata.csv',
    pmu_id='P001',
    segment_duration=15.0
)

compare_with_known_frequencies(results_1015)

quick_benchmark(
    './../data/sigId-1015.csv',
    './../data/sigId-1015-Metadata.csv', 
    results_1015,
    save_path='./../figures/sigId-1015-benchmark.png'
)

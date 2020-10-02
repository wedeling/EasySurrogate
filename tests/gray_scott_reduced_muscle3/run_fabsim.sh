cd $job_results
$run_prefix

/usr/bin/env > env.log

$muscle_manager_exec reduced.ymmsl &
python3 $gray_scott_micro_exec --muscle-instance=micro & 
python3 $gray_scott_macro_exec --muscle-instance=macro &

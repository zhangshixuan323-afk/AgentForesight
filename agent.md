agentforesight原文链接：https://arxiv.org/abs/2605.08715
任务：在sample100_by_benchmark上完成对AgentForesight-7B的测试。要求至少包含ASS，以及每一个sample在推理阶段所用的时间
环境：不在本地上跑，这个测试实际上会在远端机子上进行，配置是3张A800，各自显存81920MiB，保守估计每张卡我能用60%的显存，同名目录~/AgentForesight$ 与当前仓库同源
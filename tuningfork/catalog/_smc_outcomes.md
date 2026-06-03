- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL SMC run error: TypeError: build_sampling_algorithm.<locals>.step_fn() got an unexpected keyword argument 'inverse_mass_matrix'
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL SMC run error: TypeError: scan body function carry input and carry output must have the same pytree structure, but they differ:

The input carry component carry[1].sampler_state is a <class 'dict'> with 2 children but the corresponding component of the carry output is a <class 'dict'> with 1 child, so the numbers of children do not match, with the symmetric difference of key sets: {'step_size'}.

Revise the function so that the carry output has the same pytree structure as the carry input.
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL SMC run error: TypeError: scan body function carry input and carry output must have the same pytree structure, but they differ:

The input carry component carry[1].sampler_state is a <class 'dict'> with 2 children but the corresponding component of the carry output is a <class 'dict'> with 1 child, so the numbers of children do not match, with the symmetric difference of key sets: {'step_size'}.

Revise the function so that the carry output has the same pytree structure as the carry input.
- [logistic_synthetic/adaptive_tempered_smc/hmc] FAIL z=5.083 ess=816.5
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL SMC run error: TypeError: scan body function carry input and carry output must have the same pytree structure, but they differ:

The input carry component carry[1].sampler_state is a <class 'dict'> with 2 children but the corresponding component of the carry output is a <class 'dict'> with 1 child, so the numbers of children do not match, with the symmetric difference of key sets: {'step_size'}.

Revise the function so that the carry output has the same pytree structure as the carry input.
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL SMC run error: TypeError: scan body function carry input and carry output must have equal types, but they differ:

The input carry component carry[1].parameter_override['inverse_mass_matrix'] has type float32[1000,3] but the corresponding output carry component has type float32[3,3], so the shapes do not match.

Revise the function so that all output types match the corresponding input types.
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=2.974 ess=962.0
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=2.089 ess=971.5
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=2.742 ess=973.2
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=3.389 ess=969.7
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=2.974 ess=962.0
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=4.066 ess=956.2
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=4.246 ess=922.2
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=3.376 ess=971.2
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=2.825 ess=989.3
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=3.431 ess=966.4
- [gmm_25/adaptive_tempered_smc/rwm] FAIL SMC run error: TypeError: build_sampling_algorithm.<locals>.step_fn() got an unexpected keyword argument 'sigma'
- [logistic_synthetic/inner_kernel_tuning/hmc] FAIL z=4.276 ess=908.1
- [gmm_25/inner_kernel_tuning/hmc] FAIL z=2.059 ess=1000.0
- [gmm_25/inner_kernel_tuning/hmc] FAIL z=2.059 ess=1000.0
- [logistic_synthetic/adaptive_tempered_smc/rwm] FAIL z=4.853 ess=879.2
- [logistic_synthetic/adaptive_tempered_smc/rwm] FAIL z=2.320 ess=999.9
- [logistic_synthetic/adaptive_tempered_smc/rwm] FAIL z=4.847 ess=901.3
- [neals_funnel/inner_kernel_tuning/hmc] FAIL z=2.729 ess=1000.0
- [logistic_synthetic/adaptive_tempered_smc/rwm] FAIL z=2.608 ess=1970.8

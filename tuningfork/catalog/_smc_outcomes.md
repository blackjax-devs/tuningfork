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

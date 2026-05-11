# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions for running MCMC and SMC samplers during benchmarking.

Particle initialization and SMC runner utilities for executing
baseline configurations on the model suite.
"""

from tuningfork.runner.smc import init_particles_from_prior, run_smc

__all__ = ["init_particles_from_prior", "run_smc"]

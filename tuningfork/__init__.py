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
"""tuningfork — A BlackJAX-native benchmark library for MCMC/VI/SMC samplers.

A toolkit for systematic benchmarking of 24 base methods × 12 warmup strategies
× 6 SMC variants against a 14-model suite, with 3-tier calibration
(certified references and generated recipe evidence)
and headline metric `min-bulk-ESS / total_grad_evals`.
"""

__version__ = "0.0.0.dev0"

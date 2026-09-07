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
"""Supplied frozen coordinate transports — validation artifact.

This package holds a *supplied* (already-parameterised) coordinate chart and the
exact algebra needed to validate it.  It deliberately contains **no fitter, no
selection rule, no warmup descriptor and no execution recipe**: nothing here
learns a chart from data, and nothing here is wired into the recipe registry.

Only the convention version is public.  The chart implementation is private
(``_chart``) while the conventions it fixes are still under review.
"""

CONVENTION_VERSION = "frozen-transport-chart/v1"
"""Identifier for the coordinate and series conventions fixed by this package.

v1 fixes: clock-last coordinate order, the reflector orientation keyed on
h[-1], and the phi series threshold/order selected by measurement in
:mod:`tuningfork.transport._phi`.  Earlier exploratory implementations used different
conventions; **no bitwise equivalence with them is claimed or intended**.
"""

__all__ = ["CONVENTION_VERSION"]

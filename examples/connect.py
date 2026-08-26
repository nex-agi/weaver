# Copyright (c) Nex-AGI. All rights reserved.
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
import os

from weaver import ServiceClient


def main():
    # Pass API key directly
    # Or, you can set the environment variable WEAVER_API_KEY
    # export WEAVER_API_KEY=sk-your-api-key-here
    with ServiceClient(
        api_key=os.getenv("WEAVER_API_KEY"),
    ) as client:
        models = client.list_supported_models(detailed=True)
        for model in models:
            prices = {
                mode.display_name: {
                    kind: str(price.unit_price_usd) for kind, price in mode.prices.items()
                }
                for mode in model.training_modes
            }
            print(model.name, prices)


if __name__ == "__main__":
    main()

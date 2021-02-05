import numpy as np
import easyvvuq as uq


class CustomEncoder(uq.encoders.JinjaEncoder, encoder_name='CustomEncoder'):
    def encode(self, params={}, target_dir='', fixtures=None):

        # scale default values found in pre param file
        # mortality_factor_array = [params["Mortality_factor"]] * 21
        # params["Mortality_factor_array"] = mortality_factor_array

        proportion_symptomatic = [params["Proportion_symptomatic"]] * 17
        params["Proportion_symptomatic_array"] = proportion_symptomatic

        default_contact_rates = np.array([0.6, 0.7, 0.75, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 0.75, 0.5])
        contact_rates = default_contact_rates ** params["Relative_spatial_contact_rates_by_age_power"]
        params["Relative_spatial_contact_rates_by_age_array"] = contact_rates

        super().encode(params, target_dir, fixtures)

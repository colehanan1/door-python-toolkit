"""
Concentration-Response Models
==============================

Hill equation fitting and concentration-response curve modeling.

This module provides tools for modeling realistic concentration-response
relationships for odorant-receptor interactions.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class HillParameters:
    """
    Parameters for Hill equation dose-response curve.

    The Hill equation: R = R_max * C^n / (EC50^n + C^n)

    Attributes:
        r_max: Maximum response
        ec50: Half-maximal effective concentration
        hill_coefficient: Hill coefficient (cooperativity)
        r_baseline: Baseline response (no odor)
    """

    r_max: float
    ec50: float
    hill_coefficient: float
    r_baseline: float = 0.0

    def evaluate(self, concentrations: np.ndarray) -> np.ndarray:
        """
        Evaluate Hill equation at given concentrations.

        Args:
            concentrations: Array of concentrations

        Returns:
            Array of predicted responses
        """
        c_n = np.power(concentrations, self.hill_coefficient)
        ec50_n = np.power(self.ec50, self.hill_coefficient)
        responses = self.r_baseline + self.r_max * c_n / (ec50_n + c_n)
        return responses


class ConcentrationResponseModel:
    """
    Model concentration-response relationships for odorant-receptor pairs.

    This class fits Hill equations to DoOR data and generates concentration
    response curves for realistic neural network training data.

    Example:
        >>> model = ConcentrationResponseModel()
        >>> params = model.fit_hill_equation(
        ...     concentrations=np.array([0.001, 0.01, 0.1, 1.0]),
        ...     responses=np.array([0.1, 0.3, 0.7, 0.9])
        ... )
        >>> print(f"EC50: {params.ec50:.3f}")
    """

    @staticmethod
    def hill_equation(
        concentration: float, r_max: float, ec50: float, n: float, r_baseline: float = 0.0
    ) -> float:
        """
        Hill equation for concentration-response relationship.

        Args:
            concentration: Odorant concentration
            r_max: Maximum response
            ec50: Half-maximal effective concentration
            n: Hill coefficient (cooperativity)
            r_baseline: Baseline response

        Returns:
            Predicted response
        """
        c_n = np.power(concentration, n)
        ec50_n = np.power(ec50, n)
        return r_baseline + r_max * c_n / (ec50_n + c_n)

    def fit_hill_equation(
        self,
        concentrations: np.ndarray,
        responses: np.ndarray,
        bounds: Optional[Tuple] = None,
    ) -> HillParameters:
        """
        Fit Hill equation to concentration-response data.

        Args:
            concentrations: Array of concentrations
            responses: Array of measured responses
            bounds: Optional parameter bounds ([lower], [upper])

        Returns:
            HillParameters object with fitted parameters

        Example:
            >>> model = ConcentrationResponseModel()
            >>> concentrations = np.array([0.001, 0.01, 0.1, 1.0])
            >>> responses = np.array([0.1, 0.3, 0.7, 0.9])
            >>> params = model.fit_hill_equation(concentrations, responses)
        """
        # Remove NaN values
        mask = ~np.isnan(responses)
        concentrations = concentrations[mask]
        responses = responses[mask]

        if len(concentrations) < 3:
            logger.warning("Insufficient data points for Hill fit, using defaults")
            return HillParameters(
                r_max=responses.max() if len(responses) > 0 else 1.0,
                ec50=0.1,
                hill_coefficient=1.0,
                r_baseline=responses.min() if len(responses) > 0 else 0.0,
            )

        # Initial parameter guesses
        r_max_init = responses.max()
        r_baseline_init = responses.min()
        ec50_init = concentrations[len(concentrations) // 2]
        n_init = 1.0

        p0 = [r_max_init, ec50_init, n_init, r_baseline_init]

        # Parameter bounds
        if bounds is None:
            bounds = (
                [0, concentrations.min() * 0.1, 0.1, -1.0],  # Lower bounds
                [responses.max() * 2, concentrations.max() * 10, 10.0, responses.max()],  # Upper
            )

        try:
            popt, _ = curve_fit(
                lambda c, rmax, ec50, n, rbase: self.hill_equation(c, rmax, ec50, n, rbase),
                concentrations,
                responses,
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )

            return HillParameters(
                r_max=popt[0],
                ec50=popt[1],
                hill_coefficient=popt[2],
                r_baseline=popt[3],
            )

        except Exception as e:
            logger.warning(f"Hill fit failed: {e}, using initial estimates")
            return HillParameters(
                r_max=r_max_init,
                ec50=ec50_init,
                hill_coefficient=n_init,
                r_baseline=r_baseline_init,
            )

    def predict_concentration_response(
        self,
        params: HillParameters,
        concentrations: np.ndarray,
    ) -> np.ndarray:
        """
        Predict responses at specified concentrations.

        Args:
            params: Hill parameters
            concentrations: Array of concentrations to evaluate

        Returns:
            Array of predicted responses

        Example:
            >>> model = ConcentrationResponseModel()
            >>> params = HillParameters(r_max=1.0, ec50=0.1, hill_coefficient=1.5)
            >>> concentrations = np.logspace(-4, 0, 50)
            >>> responses = model.predict_concentration_response(params, concentrations)
        """
        return params.evaluate(concentrations)

    def generate_concentration_series(
        self,
        params: HillParameters,
        log_start: float = -4,
        log_end: float = 0,
        n_points: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate concentration series with predicted responses.

        Args:
            params: Hill parameters
            log_start: Log10 of starting concentration
            log_end: Log10 of ending concentration
            n_points: Number of concentration points

        Returns:
            Tuple of (concentrations, responses)

        Example:
            >>> model = ConcentrationResponseModel()
            >>> params = HillParameters(r_max=1.0, ec50=0.1, hill_coefficient=1.5)
            >>> conc, resp = model.generate_concentration_series(params)
            >>> plt.plot(conc, resp)
        """
        concentrations = np.logspace(log_start, log_end, n_points)
        responses = self.predict_concentration_response(params, concentrations)
        return concentrations, responses

    def model_mixture_interactions(
        self,
        component_params: List[HillParameters],
        concentrations: np.ndarray,
        interaction_type: str = "additive",
    ) -> np.ndarray:
        """
        Model odor mixture interactions.

        Args:
            component_params: List of Hill parameters for each component
            concentrations: Concentrations for each component (2D array)
            interaction_type: Type of interaction ('additive', 'competitive', 'synergistic')

        Returns:
            Array of predicted mixture responses

        Example:
            >>> model = ConcentrationResponseModel()
            >>> params1 = HillParameters(r_max=1.0, ec50=0.1, hill_coefficient=1.5)
            >>> params2 = HillParameters(r_max=0.8, ec50=0.2, hill_coefficient=1.2)
            >>> conc = np.array([[0.1, 0.2], [0.2, 0.1]])
            >>> responses = model.model_mixture_interactions([params1, params2], conc)
        """
        if concentrations.ndim == 1:
            concentrations = concentrations.reshape(-1, 1)

        n_samples = concentrations.shape[0]
        responses = np.zeros(n_samples)

        for i in range(n_samples):
            component_responses = []
            for j, params in enumerate(component_params):
                c = concentrations[i, j] if j < concentrations.shape[1] else 0.0
                r = self.predict_concentration_response(params, np.array([c]))[0]
                component_responses.append(r)

            if interaction_type == "additive":
                # Simple addition
                responses[i] = sum(component_responses)

            elif interaction_type == "competitive":
                # Components compete for receptors
                responses[i] = max(component_responses)

            elif interaction_type == "synergistic":
                # Synergistic enhancement
                responses[i] = sum(component_responses) * (1 + 0.5 * len(component_responses))

            else:
                raise ValueError(f"Unknown interaction type: {interaction_type}")

        return responses

    def fit_receptor_family(
        self,
        response_matrix: pd.DataFrame,
        receptor_name: str,
        concentration_column: str = "concentration",
    ) -> Dict[str, HillParameters]:
        """
        Fit Hill equations for all odorants for a given receptor.

        Args:
            response_matrix: DataFrame with odorant responses
            receptor_name: Name of receptor column
            concentration_column: Name of concentration column

        Returns:
            Dictionary mapping odorant to HillParameters

        Example:
            >>> model = ConcentrationResponseModel()
            >>> params_dict = model.fit_receptor_family(df, "Or42b")
            >>> for odorant, params in params_dict.items():
            ...     print(f"{odorant}: EC50={params.ec50:.3f}")
        """
        if receptor_name not in response_matrix.columns:
            raise ValueError(f"Receptor {receptor_name} not found in matrix")

        odorant_params = {}

        # Group by odorant and fit each
        odorants = response_matrix.index.unique()

        for odorant in odorants:
            odorant_data = response_matrix.loc[odorant]

            if concentration_column in odorant_data.columns:
                concentrations = odorant_data[concentration_column].values
            else:
                # Use row indices as proxy for concentration series
                concentrations = np.logspace(-3, 0, len(odorant_data))

            responses = odorant_data[receptor_name].values

            # Remove NaN
            mask = ~np.isnan(responses)
            if mask.sum() < 2:
                continue

            try:
                params = self.fit_hill_equation(
                    concentrations[mask], responses[mask]
                )
                odorant_params[odorant] = params
            except Exception as e:
                logger.warning(f"Failed to fit {odorant}: {e}")
                continue

        logger.info(
            f"Fitted Hill equations for {len(odorant_params)} odorants "
            f"for receptor {receptor_name}"
        )

        return odorant_params

    def add_concentration_noise(
        self,
        responses: np.ndarray,
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Add realistic noise to response data.

        Args:
            responses: Array of responses
            noise_type: Type of noise ('gaussian', 'poisson', 'lognormal')
            noise_level: Noise level (standard deviation or scale factor)

        Returns:
            Array of noisy responses

        Example:
            >>> model = ConcentrationResponseModel()
            >>> clean = np.array([0.5, 0.7, 0.9])
            >>> noisy = model.add_concentration_noise(clean, noise_level=0.1)
        """
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_level, size=responses.shape)
            noisy_responses = responses + noise

        elif noise_type == "poisson":
            # Scale responses to be suitable for Poisson
            scaled = responses * 100  # Assuming responses are 0-1
            noisy_scaled = np.random.poisson(scaled)
            noisy_responses = noisy_scaled / 100.0

        elif noise_type == "lognormal":
            noise = np.random.lognormal(0, noise_level, size=responses.shape)
            noisy_responses = responses * noise

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Clip to valid range
        noisy_responses = np.clip(noisy_responses, 0, None)

        return noisy_responses

    def create_dose_response_dataset(
        self,
        params_dict: Dict[str, HillParameters],
        n_concentrations: int = 20,
        add_noise: bool = True,
        noise_level: float = 0.05,
    ) -> pd.DataFrame:
        """
        Create complete dose-response dataset for multiple odorants.

        Args:
            params_dict: Dictionary mapping odorant to HillParameters
            n_concentrations: Number of concentration points per odorant
            add_noise: Whether to add realistic noise
            noise_level: Noise level if add_noise=True

        Returns:
            DataFrame with concentration-response data

        Example:
            >>> model = ConcentrationResponseModel()
            >>> params = {
            ...     "hexanol": HillParameters(1.0, 0.1, 1.5),
            ...     "ethyl butyrate": HillParameters(0.8, 0.2, 1.2)
            ... }
            >>> dataset = model.create_dose_response_dataset(params)
        """
        rows = []

        for odorant, params in params_dict.items():
            concentrations, responses = self.generate_concentration_series(
                params, n_points=n_concentrations
            )

            if add_noise:
                responses = self.add_concentration_noise(
                    responses, noise_level=noise_level
                )

            for c, r in zip(concentrations, responses):
                rows.append(
                    {
                        "odorant": odorant,
                        "concentration": c,
                        "log_concentration": np.log10(c),
                        "response": r,
                        "r_max": params.r_max,
                        "ec50": params.ec50,
                        "hill_coefficient": params.hill_coefficient,
                    }
                )

        df = pd.DataFrame(rows)
        logger.info(
            f"Created dose-response dataset: {len(df)} data points, "
            f"{len(params_dict)} odorants"
        )

        return df

import pandas as pd
import numpy as np

def generate_design_space(components, num_samples):
    """
    Generates a design space DataFrame based on material components and the number of samples.

    This is a pure logic function, free of any Streamlit UI code, making it easily testable.

    Args:
        components (list of dict): A list of component dictionaries, where each dict has
                                   'name' and 'properties' keys.
        num_samples (int): The number of sample formulations to generate.

    Returns:
        pandas.DataFrame: A DataFrame representing the generated design space, with columns
                          for component ratios and calculated properties.
    """
    if not components or not all(c.get('name') for c in components):
        # Return an empty DataFrame or raise an error if components are not well-defined
        return pd.DataFrame()

    component_names = [c['name'] for c in components]

    # Generate random ratios that sum to 1 using the Dirichlet distribution
    samples = np.random.dirichlet(np.ones(len(component_names)), size=num_samples)

    df = pd.DataFrame(samples, columns=[f"{name}_ratio" for name in component_names])

    # --- Robust Property Calculation ---
    all_property_keys = set()
    for component in components:
        all_property_keys.update(component.get('properties', {}).keys())

    for prop_name in sorted(list(all_property_keys)):
        df[prop_name] = 0
        for i, comp in enumerate(components):
            ratio_col = f"{comp['name']}_ratio"
            prop_value = comp.get('properties', {}).get(prop_name, 0)
            df[prop_name] += df[ratio_col] * prop_value

    return df

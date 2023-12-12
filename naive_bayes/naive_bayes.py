import pandas as pd
import numpy as np
from scipy.stats import norm
from colorama import Fore, Style


def set_pandas_display_options():
    pd.set_option('display.float_format', lambda x: '%.2f' % x)


def load_and_prepare_data(file_name):
    df = pd.read_csv(file_name)
    df.drop('date', axis=1, inplace=True, errors='ignore')
    return df


def calculate_stats(df):
    return df.groupby('weather').agg(['mean', 'std'])


def print_stats(stats, df):
    for weather in df['weather'].unique():
        print(Fore.GREEN + f'Weather: {weather}' + Style.RESET_ALL)
        for feature in ['precipitation', 'temp_max', 'temp_min', 'wind']:
            mean = stats.loc[weather, (feature, 'mean')]
            std = stats.loc[weather, (feature, 'std')]
            print(Fore.RED + f'Feature: {feature}' + Style.RESET_ALL)
            print(f'Mean: {mean:.2f}')
            print(f'Standard Deviation: {std:.2f}\n')


def calculate_prior(df):
    return df['weather'].value_counts() / len(df)


def calculate_posterior(new_instance, stats, prior, df):
    best_posterior = -1
    best_weather = None
    for weather in df['weather'].unique():
        print(Fore.GREEN + f'Weather: {weather}' + Style.RESET_ALL)
        likelihood = 1
        for feature in ['precipitation', 'temp_max', 'temp_min', 'wind']:
            mean = stats.loc[weather, (feature, 'mean')]
            std = stats.loc[weather, (feature, 'std')]
            if std == 0:
                if new_instance[feature] != mean:
                    likelihood = 0
                    break
            else:
                std += 1e-6
                likelihood *= norm.pdf(new_instance[feature], mean, std)
        posterior = prior[weather] * likelihood
        print(f'Original Posterior Probability: {format(posterior, ".10f")}')
        transformed_posterior = (-np.log(posterior))**-1
        print(
            f'Transformed Posterior Probability: {format(transformed_posterior, ".10f")}\n')
        if transformed_posterior > best_posterior:
            best_posterior = transformed_posterior
            best_weather = weather
    return best_weather, best_posterior


def main():
    set_pandas_display_options()
    df = load_and_prepare_data('../dataset/weather_data.csv')
    stats = calculate_stats(df)
    print_stats(stats, df)
    prior = calculate_prior(df)
    new_instance = {'precipitation': 0.0,
                    'temp_max': 12.8, 'temp_min': 5.0, 'wind': 4.7}
    best_weather, best_posterior = calculate_posterior(
        new_instance, stats, prior, df)
    print(
        f"Best posterior: {best_weather} with transformed probability {format(best_posterior, '.6f')}")


if __name__ == "__main__":
    main()

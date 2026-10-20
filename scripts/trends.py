import sys
import numpy as np
from shoreshop3_analysis import data


def main() -> int:
    obs = data.Observations()
    models = data.Models()

    obs_trends = {}

    #with open('observation-trends.csv', 'r') as file:
    #    next(file)
    #    for line in file:
    #        transect, a = line.rstrip().split(',')
    #        obs_trends[transect] = float(a)

    with open('observation-trends.csv', 'w') as file:
        print('transect,trend', file=file)
        transects = obs.get_transects()
        for i, transect in enumerate(transects):
            print(f'\r{i+1}/{len(transects)}', end='', flush=True)
            x = obs.get_single(transect)['x'].values
            t = np.arange(x.size) * coastsat.dt()
            a, b = np.polyfit(t, x, 1)
            obs_trends[transect] = a
            print(transect, a, sep=',', file=file, flush=True)

    print()

    with open('model-trends.csv', 'w') as file:
        print('model,transect,model_trend,obs_trend', file=file)
        all_models = models.get_models()
        transects = models.get_all_transects()
        i = 0
        for model in all_models:
            for transect in transects:
                i += 1
                print(f'\r{i}/{len(all_models)*len(transects)}', end='', flush=True)
                x = models.get_model_transect_data(model, transect)['x'].values
                if len(x) == 0:
                    continue
                t = np.arange(x.size)
                a, b = np.polyfit(t, x, 1)
                obs_a = obs_trends[transect]
                print(model, transect, a, obs_a, sep=',', file=file, flush=True)

    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
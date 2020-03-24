# COVID19 modelling and inference

## Installation

    git clone https://github.com/mfkasim91/idcovid19
    cd idcovid19
    python -m pip install -e .

You would need Python 3.7 or above to be able to install.

## Practical usage

To reproduce the graphics using China's data, type

    python main.py --model model2 --data cn --infer --filter r0_4

To reproduce the graphics using Indonesia's data, type

    python main.py --model model2 --data id --infer
    python main.py --model model2 --data id --infer --filter r0_4 dec_period

## Command line usage

    usage: main.py [-h] [--data DATA] [--model MODEL]
                   [--filters [FILTERS [FILTERS ...]]] [--infer] [--large]
                   [--nchains NCHAINS]

    optional arguments:
      -h, --help            show this help message and exit
      --data DATA           Which data to use (currently only 'id' or 'cn')
      --model MODEL         The mathematical model to use ('model1' or 'model2')
      --filters [FILTERS [FILTERS ...]]
                            Filter the posterior data (see
                            idcovid19/models/model*.py for filter options)
      --infer               Flag to invoke the inference mode
      --large               If invoked, then it will collect 10,000 samples
                            (ignored if no --infer).
      --nchains NCHAINS     The number of parallel chains in MCMC (ignored if no
                            --infer).    

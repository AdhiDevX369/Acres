# PredYieldProfitableOrNot

This repository contains code for predicting whether a yield will be profitable or not. 

## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/AdhiDevX369/Acres.git
    cd Acres
    ```

2. Create and activate a new Conda virtual environment:

    ```shell
    conda create --prefix .\venv python=3.8
    conda activate pred-yield
    ```

3. Install the required packages:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

1. Run the setup script:

    ```shell
    pip install .
    ```

2. Navigate to the `notebooks` > `experiment` directory:

    ```shell
    cd notebooks/experiment
    ```


3. open the `finalized.ipynb` file and run the cells.
    
    ```shell
    jupyter notebook finalized.ipynb
    ```
    
4. The model will be trained and saved in the `models` directory.

## Contributing

Contributions are welcome! Please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).

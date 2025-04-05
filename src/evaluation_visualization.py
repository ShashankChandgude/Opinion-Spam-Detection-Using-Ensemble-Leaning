import matplotlib.pyplot as plt
import pandas as pd

def plot_error_curves(train_errors: dict, test_errors: dict) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(list(train_errors.keys()), list(train_errors.values()), marker='o', label='Train Error')
    plt.plot(list(test_errors.keys()), list(test_errors.values()), marker='o', label='Test Error')
    plt.xlabel('Classifier / Ensemble')
    plt.ylabel('Error')
    plt.title('Train and Test Errors')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_results_dataframe(results_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(results_dict).T
    df.index.name = 'Classifier'
    df.reset_index(inplace=True)
    return df

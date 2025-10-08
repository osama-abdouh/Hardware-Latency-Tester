import os
import questionary
from tqdm import tqdm
from utils.model_utils import check_if_path_is_model, load_trained_model
from components.colors import colors

def single_model_test(hw_mod, hw_choose, path, results=None):
    if results is None:
        results = []

    model = load_trained_model(path)
    with tqdm(total=len(hw_choose), desc="Testing configurations", unit="test") as pbar:
        for hw_c in hw_choose:
            if hw_c in hw_mod.nvdla:
                latency = hw_mod.get_model_latency(model, hw_mod.nvdla[hw_c]['path'])
                results.append({
                    "model": path,
                    "HW config": hw_c,
                    "latency(s)": latency / (10**9)
                })
            pbar.update(1)  # Aggiorna la barra di caricamento
    return results


def multi_model_test(hw_mod, hw_choose, path, results=None):
    if results is None:
        results = []
    model_files = []

    print(colors.CYAN, f"Scanning directory: {path}", colors.ENDC)
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if check_if_path_is_model(file_path):
                model_files.append(file_path)
                
    print(colors.OKGREEN, f"Found {len(model_files)} model(s) in the directory.", colors.ENDC)

    if not model_files:
        print(colors.FAIL, "No models found in the specified directory.", colors.ENDC)
        return
    else:
        confirm = questionary.confirm(f"Proceed to test {len(model_files)} model(s)?", default=True).ask()
        if not confirm:
            print(colors.FAIL, "Operation cancelled by user.", colors.ENDC)
            return
    
    with tqdm(total=len(model_files), desc="Testing models", unit="model") as pbar:
        for file in model_files:
            relative_path = os.path.relpath(file, path)
            model = load_trained_model(file, show_info=False)
            for hw_c in hw_choose:
                if hw_c in hw_mod.nvdla:
                    latency = hw_mod.get_model_latency(model, hw_mod.nvdla[hw_c]['path'])
                    results.append({
                        "model": relative_path,
                        "HW config": hw_c,
                        "latency(s)": latency/10**9
                    })
            pbar.update(1)  # Aggiorna la barra di caricamento
    print(colors.OKGREEN, f"Testing completed. total results: {len(results)}", colors.ENDC)
    return results
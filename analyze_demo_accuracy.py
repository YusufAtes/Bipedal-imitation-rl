#!/usr/bin/env python3
"""
Script to analyze demo accuracy across all models and demo types.
Creates accuracy tables for:
1. Rotation demo: Total success rate per model
2. Noisy demo: Total success rate per model per plane
3. Velocity difference demo: MSE between measured and actual velocities
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def find_demo_files():
    """Find all demo CSV files in the project."""
    demo_files = []
    
    # Find all demo_data CSV files
    demo_data_files = glob.glob("**/demo_data_*.csv", recursive=True)
    demo_files.extend(demo_data_files)
    
    # Find simbicon CSV files
    simbicon_files = glob.glob("cma-es_simbicon/simbicon_*.csv")
    demo_files.extend(simbicon_files)
    
    return demo_files

def extract_model_info(filepath):
    """Extract model information from filepath."""
    path_parts = Path(filepath).parts
    
    # Extract model type and configuration
    if "configurations" in path_parts:
        config_idx = path_parts.index("configurations")
        if config_idx + 1 < len(path_parts):
            config_name = path_parts[config_idx + 1]
        else:
            config_name = "unknown"
        
        # Extract PPO number
        ppo_num = "unknown"
        for part in path_parts:
            if part.startswith("PPO_") or part.startswith("RecurrentPPO_"):
                ppo_num = part
                break
                
        model_type = "mlp" if "mlp" in filepath else "lstm"
        return f"{config_name}_{ppo_num}_{model_type}"
    
    elif "simbicon" in filepath:
        return "simbicon"
    else:
        return "unknown"

def analyze_rotation_demo():
    """Analyze rotation demo data and calculate success rates."""
    print("=== ROTATION DEMO ANALYSIS ===")
    print("Model\t\t\t\tSuccess Rate")
    print("-" * 50)
    
    rotation_files = [f for f in find_demo_files() if "rotation" in f]
    rotation_results = {}
    
    for filepath in rotation_files:
        try:
            df = pd.read_csv(filepath)
            if len(df) == 0:
                continue
                
            model_name = extract_model_info(filepath)
            
            # Calculate success rate
            total_trials = len(df)
            successful_trials = len(df[df['success'] == 1])
            success_rate = successful_trials / total_trials if total_trials > 0 else 0
            
            rotation_results[model_name] = success_rate
            print(f"{model_name:<30}\t{success_rate:.3f}")
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    return rotation_results

def analyze_noisy_demo():
    """Analyze noisy demo data and calculate success rates per plane."""
    print("\n=== NOISY DEMO ANALYSIS ===")
    print("Model\t\t\t\tPlane 0 Success Rate\tPlane 1 Success Rate")
    print("-" * 70)
    
    noisy_files = [f for f in find_demo_files() if "noisy" in f]
    noisy_results = {}
    
    # Group files by model
    model_files = {}
    for filepath in noisy_files:
        model_name = extract_model_info(filepath)
        if model_name not in model_files:
            model_files[model_name] = {"plane_0": None, "plane_1": None}
        
        if "_0.csv" in filepath:
            model_files[model_name]["plane_0"] = filepath
        elif "_1.csv" in filepath:
            model_files[model_name]["plane_1"] = filepath
    
    for model_name, files in model_files.items():
        plane_0_rate = 0
        plane_1_rate = 0
        
        # Analyze plane 0
        if files["plane_0"]:
            try:
                df = pd.read_csv(files["plane_0"])
                if len(df) > 0:
                    total_trials = len(df)
                    successful_trials = len(df[df['success'] == 1])
                    plane_0_rate = successful_trials / total_trials if total_trials > 0 else 0
            except Exception as e:
                print(f"Error processing {files['plane_0']}: {e}")
        
        # Analyze plane 1
        if files["plane_1"]:
            try:
                df = pd.read_csv(files["plane_1"])
                if len(df) > 0:
                    total_trials = len(df)
                    successful_trials = len(df[df['success'] == 1])
                    plane_1_rate = successful_trials / total_trials if total_trials > 0 else 0
            except Exception as e:
                print(f"Error processing {files['plane_1']}: {e}")
        
        noisy_results[model_name] = {"plane_0": plane_0_rate, "plane_1": plane_1_rate}
        print(f"{model_name:<30}\t{plane_0_rate:.3f}\t\t\t{plane_1_rate:.3f}")
    
    return noisy_results

def analyze_velocity_difference_demo():
    """Analyze velocity difference demo and calculate MSE."""
    print("\n=== VELOCITY DIFFERENCE DEMO ANALYSIS ===")
    print("Model\t\t\t\tMSE (Measured vs Actual)")
    print("-" * 50)
    
    vel_diff_files = [f for f in find_demo_files() if "vel_diff" in f]
    vel_diff_results = {}
    
    for filepath in vel_diff_files:
        try:
            df = pd.read_csv(filepath)
            if len(df) == 0:
                continue
                
            model_name = extract_model_info(filepath)
            
            # Calculate MSE between commanded speed and actual speed
            commanded_speeds = df['cmd speed'].values
            actual_speeds = df['mean speed'].values
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(commanded_speeds) | np.isnan(actual_speeds))
            commanded_speeds = commanded_speeds[valid_mask]
            actual_speeds = actual_speeds[valid_mask]
            
            if len(commanded_speeds) > 0:
                mse = np.mean((commanded_speeds - actual_speeds) ** 2)
                vel_diff_results[model_name] = mse
                print(f"{model_name:<30}\t{mse:.6f}")
            else:
                print(f"{model_name:<30}\tNo valid data")
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    return vel_diff_results

def create_summary_table(rotation_results, noisy_results, vel_diff_results):
    """Create a comprehensive summary table."""
    print("\n" + "="*80)
    print("COMPREHENSIVE ACCURACY SUMMARY")
    print("="*80)
    
    # Get all unique models
    all_models = set(rotation_results.keys()) | set(noisy_results.keys()) | set(vel_diff_results.keys())
    
    print(f"{'Model':<30} {'Rotation':<12} {'Noisy P0':<12} {'Noisy P1':<12} {'Vel MSE':<12}")
    print("-" * 80)
    
    for model in sorted(all_models):
        rotation_rate = rotation_results.get(model, "N/A")
        noisy_p0 = noisy_results.get(model, {}).get("plane_0", "N/A")
        noisy_p1 = noisy_results.get(model, {}).get("plane_1", "N/A")
        vel_mse = vel_diff_results.get(model, "N/A")
        
        # Format values
        rotation_str = f"{rotation_rate:.3f}" if isinstance(rotation_rate, (int, float)) else str(rotation_rate)
        noisy_p0_str = f"{noisy_p0:.3f}" if isinstance(noisy_p0, (int, float)) else str(noisy_p0)
        noisy_p1_str = f"{noisy_p1:.3f}" if isinstance(noisy_p1, (int, float)) else str(noisy_p1)
        vel_mse_str = f"{vel_mse:.6f}" if isinstance(vel_mse, (int, float)) else str(vel_mse)
        
        print(f"{model:<30} {rotation_str:<12} {noisy_p0_str:<12} {noisy_p1_str:<12} {vel_mse_str:<12}")

def main():
    """Main function to run all analyses."""
    print("Analyzing demo accuracy across all models...")
    print("="*60)
    
    # Change to the project directory
    os.chdir("/home/baran/Bipedal-imitation-rl")
    
    # Run analyses
    rotation_results = analyze_rotation_demo()
    noisy_results = analyze_noisy_demo()
    vel_diff_results = analyze_velocity_difference_demo()
    
    # Create summary table
    create_summary_table(rotation_results, noisy_results, vel_diff_results)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    
    # Save results to files for further use
    results = {
        'rotation': rotation_results,
        'noisy': noisy_results,
        'velocity_difference': vel_diff_results
    }
    
    # Save as JSON for programmatic access
    import json
    with open('demo_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to demo_accuracy_results.json")

if __name__ == "__main__":
    main()




import os
import shutil
import glob
import subprocess # Use subprocess to launch replay.py

# --- Configuration ---
LOGS_DIR = "traj_logs"
JUNK_DIR = os.path.join(LOGS_DIR, "junk")
GOOD_DIR = os.path.join(LOGS_DIR, "good")
XML_PATH = "unitree_robots/h1_2/avoid_h12.xml" # Use your XML file name

def run_replay(xml_path, csv_path, loop_mode):
    """
    Launches replay.py as a subprocess.
    """
    print(f"\n--- Replaying {os.path.basename(csv_path)} ---")
    
    command = [
        "python3", "replay.py",
        "--xml", xml_path,
        "--csv", csv_path,
        "--play_rate", "0.8",
        "--replay_logged_obst" 
    ]
    
    # If the file should loop for replay, add the --loop flag
    if loop_mode:
        command.append("--loop")
    
    try:
        # Run the subprocess and wait for it to finish (i.e., the user closes the viewer)
        process = subprocess.Popen(command)
        process.wait()
    except Exception as e:
        print(f"Error running replay.py: {e}")
        return False
    return True

def interactive_annotate():
    """
    Main loop for interactive data annotation.
    """
    if not os.path.exists(JUNK_DIR):
        os.makedirs(JUNK_DIR)
        print(f"Created junk directory: {JUNK_DIR}")
        
    if not os.path.exists(GOOD_DIR):
        os.makedirs(GOOD_DIR)
        print(f"Created junk directory: {GOOD_DIR}")

    csv_files = glob.glob(os.path.join(LOGS_DIR, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {LOGS_DIR}. Exiting.")
        return

    print(f"Found {len(csv_files)} trajectories to annotate.")
    
    for i, csv_file in enumerate(csv_files):
        file_name = os.path.basename(csv_file)
        
        # --- Step 1: Replay the trajectory once without looping ---
        run_replay(XML_PATH, csv_file, loop_mode=False) 
        
        # --- Step 2: Get user input ---
        while True:
            response = input("Keep this trajectory? (Y/n/r=replay): ").lower()
            
            if response in ['y', '']:
                print(f"Keeping {file_name}.")
                dest_path = os.path.join(GOOD_DIR, file_name)
                shutil.move(csv_file, dest_path)
                print(f"Moved {file_name} to good.")
                break 
                
            elif response == 'n':
                dest_path = os.path.join(JUNK_DIR, file_name)
                shutil.move(csv_file, dest_path)
                print(f"Moved {file_name} to junk.")
                break
                
            elif response == 'r':
                print(f"Launching replay of {file_name}. Close the viewer to continue...")
                # Launch the replay with the --loop flag
                run_replay(XML_PATH, csv_file, loop_mode=True)
                # After the user closes the looping viewer, prompt again
                continue 
            
            else:
                print("Invalid input. Please enter 'y', 'n', or 'r'.")

    print("\nAnnotation complete!")

if __name__ == "__main__":
    interactive_annotate()
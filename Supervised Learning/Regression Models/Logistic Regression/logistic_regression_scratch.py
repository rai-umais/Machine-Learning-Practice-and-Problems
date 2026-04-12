import numpy as np
import pandas as pd
import time
import sys
import matplotlib.pyplot as plt

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def predict_value(theta,x):
    z = np.dot(x,theta)
    return sigmoid(z)

def gradient_ascent(theta,x,y,alpha):
    m = len(y)
    for _ in range(1000):
        y_hat = predict_value(theta,x)
        error = y - y_hat

        gradient = 1/m * np.dot(x.T,error)
        theta = theta + alpha * gradient

    return theta

if __name__ == "__main__":
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # 1. Fascinating Intro
    print(f"{BLUE}{BOLD}" + "="*50)
    print("      L O G I S T I C   R E G R E S S I O N   v1.0")
    print("="*50 + f"{RESET}")

    print(f"{CYAN}Initializing Neural Engine...{RESET}")
    print(f"{CYAN}Loading 'Age vs Type' dataset...{RESET}")
    # Your Data
    df = pd.DataFrame(data=[
    [34, 78, 0],[30, 43, 0],[35, 72, 0],[60, 86, 1],[79, 75, 1],
    [75, 68, 1],[76, 87, 1],[84, 43, 0],[95, 72, 1],[67, 56, 1],
    [61, 96, 1],[88, 99, 1],[54, 43, 0],[47, 53, 0],[74, 60, 1],
    [65, 66, 1],[69, 97, 1],[70, 55, 1],[50, 50, 0],[80, 80, 1]
    ], columns=['Exam1','Exam2','Admitted'])

    X = df[['Exam1','Exam2']].values  # Now X contains 2 columns
    y = df['Admitted'].values
    x_mean = X.mean(axis = 0)   # axis = 0 tells that for each row, compute the mean/std dev downwards i.e the final
    x_std_dev = X.std(axis = 0)  #   output array should have the mean/std of each column at an index
    x_scaled = (X-x_mean)/x_std_dev
                        # i faced error that 
    theeta_0 = 0
    theeta_1 = 0
    theeta_2 = 0
    theta = np.array([theeta_0,theeta_1,theeta_2])

# BEAUTIFICATION BY GPT
    print(f"\n{YELLOW}Training in progress...{RESET}")
    for i in range(10):
        time.sleep(0.1)
        sys.stdout.write(f"{RED}{BOLD}\r[{'#' * (i+1)}{' ' * (9-i)}]{RESET} { (i+1)*10}% Complete")
        sys.stdout.flush()
    print("\n")
    # Train
    X_final = np.c_[np.ones(len(x_scaled)),x_scaled]
    theta = gradient_ascent(theta,X_final,y,0.01)
    print(f"{GREEN}Model Trained Successfully!{RESET}")
    print(f"{BOLD}Final Bias (θ0):{RESET} {theta[0]:.4f}")
    print(f"{BOLD}Final Weight (θ1):{RESET} {theta[1]:.4f}")
    print(f"{BOLD}Final Weight (θ2):{RESET} {theta[2]:.4f}")

    print("-" * 50)
    # Interactive Input
    # BEAUTIFICATION END
    try:
        val1 = input(f"\n{YELLOW}Enter your marks in Exam 1: {RESET}")
        inp1 = float(val1)
        
        val2 = input(f"\n{YELLOW}Enter your marks in Exam 2: {RESET}")
        inp2 = float(val2)
        inp_arr = np.array([inp1,inp2])         # i faced error:() after np.array
        inp_scaled = (inp_arr - x_mean) / x_std_dev    # i faced error: 


        inp_vec = np.array([1,inp_scaled[0],inp_scaled[1]])  # i faced error: The predictor now takes X with 3 indexes
        y_pred = predict_value(theta,inp_vec)


        print(f"\n{BLUE}--- PREDICTION RESULT ---{RESET}")
        print(f"For Exam 1 marks: {BOLD}{inp_arr[0]}{RESET}")
        print(f"For Exam 2 marks: {BOLD}{inp_arr[1]}{RESET}")

        if y_pred >= 0.5:
            print(f"Predicted Accepted / Rejected: {GREEN}{BOLD} Accepted {RESET}")
            print(f"Models Confiedence:         {YELLOW}{BOLD} {(y_pred * 100):.3f} {RESET} %")
        else:
            print(f"Predicted Accepted / Rejected: {RED}{BOLD} Rejected {RESET}")
            print(f"Models Confiedence: {YELLOW}{BOLD} {((1-y_pred) * 100):.3f} {RESET} %")

        print(f"{BLUE}" + "-" * 25 + f"{RESET}\n")

    except ValueError:
        print(f"{BOLD}\033[91mError: Please enter a valid numerical value(s).{RESET}")

# scatter plot
plt.scatter(X[:,0], X[:,1], c=y)

# create line
x1_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)

# scale x1
x1_scaled = (x1_vals - x_mean[0]) / x_std_dev[0]

# compute x2 (scaled)
x2_scaled = -(theta[0] + theta[1]*x1_scaled) / theta[2]

# convert back to original scale
x2_vals = x2_scaled * x_std_dev[1] + x_mean[1]

plt.plot(x1_vals, x2_vals)

plt.xlabel("Exam 1")
plt.ylabel("Exam 2")
plt.title("Decision Boundary")
plt.show()


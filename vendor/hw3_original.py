import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self, 
                 num_thresholds, 
                 genuine_scores, 
                 impostor_scores, 
                 plot_title, 
                 epsilon=1e-12):
        """
        Initialize the Evaluator object.

        Parameters:
        - num_thresholds (int): Number of thresholds to evaluate.
        - genuine_scores (array-like): Genuine scores for evaluation.
        - impostor_scores (array-like): Impostor scores for evaluation.
        - plot_title (str): Title for the evaluation plots.
        - epsilon (float): A small value to prevent division by zero.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.

        Returns:
        - float: The calculated d' value.
        """
        x = abs(np.mean(self.genuine_scores) - np.mean(self.impostor_scores))
        y = np.sqrt((np.var(self.genuine_scores) +
                     np.var(self.impostor_scores)) / 2)
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores.
        """
        plt.figure()
        
        # Plot the histogram for genuine scores
        plt.hist(
            self.genuine_scores,
            bins=np.linspace(0, 1, 31),
            color='green',
            lw=1.5,
            histtype='step',
            hatch='//',
            label='Genuine Scores' 
        )
        
        # Plot the histogram for impostor scores
        plt.hist(
            self.impostor_scores,
            bins=np.linspace(0, 1, 31),
            color='red',
            lw=1.5,
            histtype='step',
            hatch='\\',
            label='Impostor Scores' 
        )
        
        # Set the x-axis limit to ensure the histogram fits within the correct range
        plt.xlim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Step 6: Add legend to the upper left corner with a specified font size
        plt.legend(
            loc='upper left',
            fontsize=10
        )
        
        # Step 7: Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            'Matching Score',
            fontsize=12,
            weight='bold'
        )
        
        plt.ylabel(
            'Frequency',
            fontsize=12,
            weight='bold'
        )
        
        # Step 8: Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Step 9: Set font size for x and y-axis ticks
        plt.xticks(
            fontsize=10
        )
        
        plt.yticks(
            fontsize=10
        )
        
        # Step 10: Add a title to the plot with d-prime value and system title
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' % 
                  (self.get_dprime(), 
                   self.plot_title),
                  fontsize=15,
                  weight='bold')
        
        # Save the figure before displaying it
        plt.savefig('score_distribution_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Display the plot after saving
        plt.show()
        
        # Close the figure to free up resources
        plt.close()

        return

    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
    
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - FNR (list or array-like): False Negative Rate values.
    
        Returns:
        - tuple: Approximate EER and its evaluated threshold.
        """
        EER = 0
        
        # Use the evaluated threshold where FPR and FNR are closest.
        # With a finite threshold grid, their average approximates the EER.
        index = np.argmin(np.abs(np.array(FPR) - np.array(FNR)))
        EER = (FPR[index] + FNR[index]) / 2
        EER_threshold = self.thresholds[index]
        
        return EER, EER_threshold

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """
        
        # Calculate the Equal Error Rate (EER) using the get_EER method
        EER, EER_threshold = self.get_EER(FPR, FNR)
        
        # Create a new figure for plotting
        plt.figure()
        
        # Plot the Detection Error Tradeoff Curve
        plt.plot(
            FPR, FNR,
            lw=2,
            color='blue'
        )
        
        # Add a text annotation for the EER point on the curve
        # Plot the diagonal line representing equal false-positive and false-negative rates
        # Scatter plot to highlight the EER point on the curve

        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        index = np.argmin(np.abs(self.thresholds - EER_threshold))
        plt.scatter([FPR[index]], [FNR[index]], c="black", s=100)
        
        # Set the x and y-axis limits to ensure the plot fits within the range 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(
            color='gray',
            linestyle='--',
            linewidth=0.5
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            'False Pos. Rate',
            fontsize=12,
            weight='bold'
        )
        
        plt.ylabel(
            'False Neg. Rate',
            fontsize=12,
            weight='bold'
        )
        
        # Step 11: Add a title to the plot with EER value and system title
        plt.title(
            'Detection Error Tradeoff Curve \nEER = %.5f at t=%.3f\nSystem %s' %
            (EER, EER_threshold, self.plot_title),
            fontsize=15,
            weight='bold' 
        )
        
        # Step 12: Set font size for x and y-axis ticks
        plt.xticks(
            fontsize=10
        )
        
        plt.yticks(
            fontsize=10
        )
        
        # Step 13: Save the plot as an image file
        plt.savefig(
            'det_curve_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight"
        )
        
        # Step 14: Display the plot
        plt.show()
        
        # Step 15: Close the plot to free up resources
        plt.close()
    
        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """
        
        # Create a new figure for the ROC curve
        plt.figure()

        # Plot the ROC curve using FPR and TPR with specified attributes
        plt.plot(FPR, TPR, lw=2, color='blue')

        # Set x and y axis limits, add grid, and remove top and right spines
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Set labels for x and y axes, and add a title
        plt.xlabel('False Pos. Rate', fontsize=12, weight='bold')
        plt.ylabel('True Pos. Rate', fontsize=12, weight='bold')
        plt.title('Receiver Operating Characteristic Curve\nSystem %s' %
                  self.plot_title, fontsize=15, weight='bold')

        # Set font sizes for ticks, x and y labels
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Save the plot as a PNG file and display it
        plt.savefig('roc_curve_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        plt.show()

        # Close the figure to free up resources
        plt.close()
 
        return

    def compute_rates(self):
        # Initialize lists for False Positive Rate (FPR), False Negative Rate (FNR), and True Positive Rate (TPR)
        FPR, FNR, TPR = [], [], []
        # Iterate through threshold values and calculate TP, FP, TN, and FN for each threshold
        for threshold in self.thresholds:
            # Accept a match when its similarity score is >= the threshold.
            TP = np.sum(self.genuine_scores >= threshold)
            FN = np.sum(self.genuine_scores < threshold)
            FP = np.sum(self.impostor_scores >= threshold)
            TN = np.sum(self.impostor_scores < threshold)

            # Calculate FPR, FNR, and TPR based on the obtained values
            # Append calculated rates to their respective lists
            FPR.append(FP / (FP + TN))
            FNR.append(FN / (TP + FN))
            TPR.append(TP / (TP + FN))

        # Return the lists of FPR, FNR, and TPR
        return FPR, FNR, TPR

def main():
    
    seed = 42
    rng = np.random.default_rng(seed=seed)

    # Name the systems A, B, and C
    systems = ['A', 'B', 'C']
    
    for system in systems:
        # Randomly determine the number of genuine and impostor scores
        gSize = rng.integers(5000, 10000)
        iSize = rng.integers(15000, 20000)
        
        # Randomly determine how many distributions to mix
        gDists = rng.integers(1, 10)
        iDists = rng.integers(1, 10)
    
        # Generate genuine scores with multiple distribution peaks
        genuine_scores = []
        for gDist in range(gDists):  # start from 1 to gDists-1
            size = int(gSize / gDists)
            loc = (1.00 - 0.4) * rng.random() + 0.4
            scale = (0.3 - 0.05) * rng.random() + 0.05
            g_scores = np.clip(rng.normal(loc=loc, scale=scale, size=size), 0, 1)
            genuine_scores = np.concatenate([genuine_scores, g_scores])
        
        # Generate impostor scores with multiple distributions
        impostor_scores = []
        for iDist in range(iDists):  # start from 1 to iDists-1
            size = int(iSize / iDists)
            loc = (0.6 - 0.0) * rng.random() + 0.0
            scale = (0.3 - 0.05) * rng.random() + 0.05
            i_scores = np.clip(rng.normal(loc=loc, scale=scale, size=size), 0, 1)
            impostor_scores = np.concatenate([impostor_scores, i_scores])
        
        # Creating an instance of the Evaluator class
        evaluator = Evaluator(
            epsilon=1e-12,
            num_thresholds=500,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            plot_title="%s" % system
        )
        
        # Generate the FPR, FNR, and TPR using 500 threshold values equally spaced
        # between -0.1 and 1.1.
        FPR, FNR, TPR = evaluator.compute_rates()
    
        # Plot the score distribution. Include the d-prime value in the plot’s 
        # title. Your genuine scores should be green, and your impostor scores 
        # should be red. Set the x axis limits from -0.05 to 1.05
        evaluator.plot_score_distribution()
                
        # Plot the DET curve and include the EER in the plot’s title. 
        # Set the x and y axes limits from -0.05 to 1.05.
        evaluator.plot_det_curve(FPR, FNR)
        
        # Plot the ROC curve. Set the x and y axes limits from -0.05 to 1.05.
        evaluator.plot_roc_curve(FPR, TPR)
        
        
if __name__ == "__main__":
    main()


"""Bundled HW3 evaluator and its standalone synthetic-score demonstration."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class Evaluator:
    """Evaluate genuine and impostor similarity scores across a fixed threshold grid.
    """

    def __init__(self,
                 num_thresholds,
                 genuine_scores,
                 impostor_scores,
                 plot_title,
                 epsilon=1e-12):
        """Store score classes and construct evenly spaced thresholds from -0.1 to 1.1.

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
        """Compute absolute mean score separation divided by pooled standard deviation.

        Returns:
        - float: The calculated d' value.
        """
        x = abs(np.mean(self.genuine_scores) - np.mean(self.impostor_scores))
        y = np.sqrt((np.var(self.genuine_scores) +
                     np.var(self.impostor_scores)) / 2)
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """Save and display genuine and impostor score histograms with d-prime.
        """
        plt.figure()

        # Genuine and impostor histograms share bin boundaries for comparison.
        plt.hist(
            self.genuine_scores,
            bins=np.linspace(0, 1, 31),
            color='green',
            lw=1.5,
            histtype='step',
            hatch='//',
            label='Genuine Scores'
        )

        # Impostor scores are shown separately from genuine scores.
        plt.hist(
            self.impostor_scores,
            bins=np.linspace(0, 1, 31),
            color='red',
            lw=1.5,
            histtype='step',
            hatch='\\',
            label='Impostor Scores'
        )

        # Axis limits include a margin around the normalized score range.
        plt.xlim([-0.05, 1.05])

        # Grid lines support comparisons across error rates.
        plt.grid(color='gray', linestyle='--', linewidth=0.5)

        # The legend identifies genuine and impostor score distributions.
        plt.legend(
            loc='upper left',
            fontsize=10
        )

        # Axis labels identify matching scores and observation counts.
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

        # Only the left and bottom axes remain visible.
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Tick labels use a consistent size across evaluation plots.
        plt.xticks(
            fontsize=10
        )

        plt.yticks(
            fontsize=10
        )

        # The title records score separation and the evaluated system.
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' %
                  (self.get_dprime(),
                   self.plot_title),
                  fontsize=15,
                  weight='bold')

        # The score-distribution figure is saved at publication resolution.
        plt.savefig('score_distribution_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")

        # Display uses the active Matplotlib backend.
        plt.show()

        # The figure is closed after output to release plotting resources.
        plt.close()

        return

    def get_EER(self, FPR, FNR):
        """Approximate EER at the evaluated threshold with the smallest FAR/FRR difference.

        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - FNR (list or array-like): False Negative Rate values.

        Returns:
        - tuple: Approximate EER and its evaluated threshold.
        """
        EER = 0

        # The nearest FAR/FRR crossing on the evaluated grid defines the EER estimate.
        # The two error rates are averaged because the finite grid may not contain equality.
        index = np.argmin(np.abs(np.array(FPR) - np.array(FNR)))
        EER = (FPR[index] + FNR[index]) / 2
        EER_threshold = self.thresholds[index]

        return EER, EER_threshold

    def plot_det_curve(self, FPR, FNR):
        """Save and display the DET curve on linear error-rate axes.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """

        # The HW3 EER estimate and threshold identify the annotated operating point.
        EER, EER_threshold = self.get_EER(FPR, FNR)

        # The DET curve uses a separate figure.
        plt.figure()

        # The DET curve shows false rejection against false acceptance on linear axes.
        plt.plot(
            FPR, FNR,
            lw=2,
            color='blue'
        )

        # The diagonal marks equal error rates; the marker identifies the sampled crossing.



        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        index = np.argmin(np.abs(self.thresholds - EER_threshold))
        plt.scatter([FPR[index]], [FNR[index]], c="black", s=100)

        # Both axes cover the full error-rate range with a small margin.
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        # Grid lines support comparisons across error rates.
        plt.grid(
            color='gray',
            linestyle='--',
            linewidth=0.5
        )

        # Only the left and bottom axes remain visible.
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Axis labels identify false acceptance and false rejection rates.
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

        # The title records the EER estimate, sampled threshold, and system.
        plt.title(
            'Detection Error Tradeoff Curve \nEER = %.5f at t=%.3f\nSystem %s' %
            (EER, EER_threshold, self.plot_title),
            fontsize=15,
            weight='bold'
        )

        # Tick labels match the other evaluation figures.
        plt.xticks(
            fontsize=10
        )

        plt.yticks(
            fontsize=10
        )

        # The DET figure is saved at publication resolution.
        plt.savefig(
            'det_curve_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight"
        )

        # Display uses the active Matplotlib backend.
        plt.show()

        # The figure is closed after output to release plotting resources.
        plt.close()

        return

    def plot_roc_curve(self, FPR, TPR):
        """Save and display genuine acceptance against false acceptance.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """

        # The ROC curve uses a separate figure.
        plt.figure()

        # The ROC curve shows genuine acceptance against false acceptance.
        plt.plot(FPR, TPR, lw=2, color='blue')

        # Axis limits and styling match the DET figure for comparison.
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Labels identify acceptance rates and the evaluated system.
        plt.xlabel('False Pos. Rate', fontsize=12, weight='bold')
        plt.ylabel('True Pos. Rate', fontsize=12, weight='bold')
        plt.title('Receiver Operating Characteristic Curve\nSystem %s' %
                  self.plot_title, fontsize=15, weight='bold')

        # Tick labels match the other evaluation figures.
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # The ROC figure is saved before display through the active backend.
        plt.savefig('roc_curve_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        plt.show()

        # The figure is closed after output to release plotting resources.
        plt.close()

        return

    def compute_rates(self):
        """Return FAR, FRR, and TPR arrays for the configured similarity thresholds."""
        # Rate arrays follow the order of the evaluated thresholds.
        FPR, FNR, TPR = [], [], []
        # Each threshold partitions genuine and impostor scores into acceptance counts.
        for threshold in self.thresholds:
            # Scores equal to or above the threshold are accepted as matches.
            TP = np.sum(self.genuine_scores >= threshold)
            FN = np.sum(self.genuine_scores < threshold)
            FP = np.sum(self.impostor_scores >= threshold)
            TN = np.sum(self.impostor_scores < threshold)

            # False acceptance, false rejection, and genuine acceptance use class-specific totals.

            FPR.append(FP / (FP + TN))
            FNR.append(FN / (TP + FN))
            TPR.append(TP / (TP + FN))

        # FPR and FNR correspond to biometric FAR and FRR, respectively.
        return FPR, FNR, TPR

def main():
    """Run the standalone synthetic-score demonstration, separate from RQ experiments."""

    seed = 42
    rng = np.random.default_rng(seed=seed)

    # The standalone HW3 demonstration evaluates three synthetic score systems.
    systems = ['A', 'B', 'C']

    for system in systems:
        # Seeded draws set the genuine and impostor sample counts.
        gSize = rng.integers(5000, 10000)
        iSize = rng.integers(15000, 20000)

        # Each score class combines a randomly selected number of Gaussian components.
        gDists = rng.integers(1, 10)
        iDists = rng.integers(1, 10)

        # Genuine scores form a clipped Gaussian mixture in [0, 1].
        genuine_scores = []
        for gDist in range(gDists):  # One component is generated for each index from zero to gDists minus one.
            size = int(gSize / gDists)
            loc = (1.00 - 0.4) * rng.random() + 0.4
            scale = (0.3 - 0.05) * rng.random() + 0.05
            g_scores = np.clip(rng.normal(loc=loc, scale=scale, size=size), 0, 1)
            genuine_scores = np.concatenate([genuine_scores, g_scores])

        # Impostor scores form a separate clipped Gaussian mixture in [0, 1].
        impostor_scores = []
        for iDist in range(iDists):  # One component is generated for each index from zero to iDists minus one.
            size = int(iSize / iDists)
            loc = (0.6 - 0.0) * rng.random() + 0.0
            scale = (0.3 - 0.05) * rng.random() + 0.05
            i_scores = np.clip(rng.normal(loc=loc, scale=scale, size=size), 0, 1)
            impostor_scores = np.concatenate([impostor_scores, i_scores])

        # The demonstration passes synthetic scores to the same evaluation interface.
        evaluator = Evaluator(
            epsilon=1e-12,
            num_thresholds=500,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            plot_title="%s" % system
        )

        # The demonstration evaluates 500 thresholds between -0.1 and 1.1.

        FPR, FNR, TPR = evaluator.compute_rates()

        # Score histograms include d-prime and distinguish the two score classes.


        evaluator.plot_score_distribution()

        # The DET figure reports EER at the nearest sampled crossing.

        evaluator.plot_det_curve(FPR, FNR)

        # The ROC figure summarizes genuine acceptance across false-acceptance rates.
        evaluator.plot_roc_curve(FPR, TPR)


if __name__ == "__main__":
    main()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from blue_blob_images_float import image_to_float
from plotting_random_images import plot_random_images

class ModelEvaluator:
    def __init__(self, model, threshold: float = 0.5):
        """
        model: a compiled keras Model that outputs a single sigmoid probability.
        threshold: cutoff above which you call a sample “positive” (blue blob).
        """
        self.model = model
        self.threshold = threshold

    def load_data(self, blob_dir: str, galaxy_dir: str):
        # assumes image_to_float(...) returns an array of floats
        self.blob_imgs  = image_to_float(blob_dir)
        self.galaxy_imgs = image_to_float(galaxy_dir)

        print(f"Galaxy images shape: {self.galaxy_imgs.shape}, dtype: {self.galaxy_imgs.dtype}")
        print(f"Blue-blob images shape: {self.blob_imgs.shape}, dtype: {self.blob_imgs.dtype}")

        self.X_test = np.concatenate([self.galaxy_imgs, self.blob_imgs], axis=0)
        y_galaxy = np.ones(self.galaxy_imgs.shape[0], dtype=np.int64)
        y_blob   = np.zeros(self.blob_imgs.shape[0],   dtype=np.int64)
        self.y_test = np.concatenate([y_galaxy, y_blob], axis=0)

        print(f"Total test samples: {self.X_test.shape[0]}")

    def predict_and_report(self, X_test, y_test):

        # run model
        probs = self.model.predict(X_test)
        self.y_pred = (probs > self.threshold).astype(int)

        # confusion matrix
        cm = confusion_matrix(y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Blue Blob', 'Background/Galaxy'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

        # classification report
        print(classification_report(
            y_test,
            self.y_pred,
            target_names=['Blue Blob', 'Background/Galaxy']
        ))

    def _plot_misclassified(self, true_lbl, pred_lbl, num_to_plot=10, min_separation=20, title_prefix=""):
        idxs = np.where((self.y_test == true_lbl) & (self.y_pred.flatten() == pred_lbl))[0]
        if len(idxs)==0:
            print(f"No samples with true={true_lbl} & pred={pred_lbl}")
            return
        imgs = self.X_test[idxs]
        print(f"{len(idxs)} misclassified samples (true={true_lbl}→pred={pred_lbl})")
        plot_random_images(
            imgs,
            num_to_plot=num_to_plot,
            min_separation=min_separation,
            title_prefix=title_prefix
        )

    def plot_misclassified_blue_blobs(self, **kwargs):
        self._plot_misclassified(
            true_lbl=0, pred_lbl=1,
            title_prefix="Misclassified Blue Blob Images",
            **kwargs
        )

    def plot_misclassified_galaxies(self, **kwargs):
        self._plot_misclassified(
            true_lbl=1, pred_lbl=0,
            title_prefix="Misclassified Galaxy Images",
            **kwargs
        )

    def evaluate(
        self,
        blob_dir: str,
        galaxy_dir: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_to_plot: int = 10,
        min_separation: int = 20
    ):
        """
        Runs the full pipeline:
         1) load data from blob_dir & galaxy_dir
         2) predict & show confusion matrix + report
         3) plot misclassified blue blobs & galaxies
        """
        self.load_data(blob_dir, galaxy_dir)
        self.predict_and_report(X_test, y_test)
        self.plot_misclassified_blue_blobs(num_to_plot=num_to_plot, min_separation=min_separation)
        self.plot_misclassified_galaxies(num_to_plot=num_to_plot, min_separation=min_separation)
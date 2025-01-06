from typing import List, Optional, Tuple, Dict, Union

from PIL import Image
import pydicom
from presidio_analyzer import PatternRecognizer

from presidio_image_redactor import (
    OCR,
    BboxProcessor,
    DicomImageRedactorEngine,
    ImageAnalyzerEngine,
    ImagePiiVerifyEngine,
    TesseractOCR,
)
from presidio_image_redactor.entities import ImageRecognizerResult


class DicomImagePiiVerifyEngine(ImagePiiVerifyEngine, DicomImageRedactorEngine):
    """Class to handle verification and evaluation for DICOM de-identification."""

    def __init__(
        self,
        ocr_engine: Optional[OCR] = None,
        image_analyzer_engine: Optional[ImageAnalyzerEngine] = None,
    ):
        """Initialize DicomImagePiiVerifyEngine object.

        :param ocr_engine: OCR engine to use.
        :param image_analyzer_engine: Image analyzer engine to use.
        """
        # Initialize OCR engine
        if not ocr_engine:
            self.ocr_engine = TesseractOCR()
        else:
            self.ocr_engine = ocr_engine

        # Initialize image analyzer engine
        if not image_analyzer_engine:
            self.image_analyzer_engine = ImageAnalyzerEngine()
        else:
            self.image_analyzer_engine = image_analyzer_engine

        # Initialize bbox processor
        self.bbox_processor = BboxProcessor()

    def verify_dicom_instance(
        self,
        instance: pydicom.dataset.FileDataset,
        padding_width: int = 25,
        display_image: bool = True,
        show_text_annotation: bool = True,
        use_metadata: bool = True,
        ocr_kwargs: Optional[dict] = None,
        ad_hoc_recognizers: Optional[List[PatternRecognizer]] = None,
        **text_analyzer_kwargs,
    ) -> Tuple[
        Optional[Image.Image],
        Dict[str, Union[str, int, float]],
        List[ImageRecognizerResult],
    ]:
        """Verify PII on a single DICOM instance.

        :param instance: Loaded DICOM instance including pixel data and metadata.
        :param padding_width: Padding width to use when running OCR.
        :param display_image: If the verification image is displayed and returned.
        :param show_text_annotation: True to display entity type when displaying
        image with bounding boxes.
        :param use_metadata: Whether to redact text in the image that
        are present in the metadata.
        :param ocr_kwargs: Additional params for OCR methods.
        :param ad_hoc_recognizers: List of PatternRecognizer objects to use
        for ad-hoc recognizer.
        :param text_analyzer_kwargs: Additional values for the analyze method
        in ImageAnalyzerEngine.

        :return: Image with boxes identifying PHI, OCR results,
        and analyzer results.
        """
        # Load image
        image = self._get_image_from_dicom(instance, padding_width)
        is_greyscale = self._check_if_greyscale(instance)

        # Get OCR results
        analyzer_results, ocr_result, preprocessed_image, preprocessing_metadata = (
            self.image_analyzer_engine.analyze_and_return_intermediate(
                image, ocr_kwargs, **text_analyzer_kwargs
            )
        )

        ocr_bboxes = self.bbox_processor.get_bboxes_from_ocr_results(ocr_result)

        # Prepare for plotting
        pii_bboxes = self.image_analyzer_engine.get_pii_bboxes(
            ocr_bboxes, analyzer_results
        )
        if is_greyscale:
            use_greyscale_cmap = True
        else:
            use_greyscale_cmap = False

        # Get image with verification boxes
        verify_image: Image = (
            self.image_analyzer_engine.add_custom_bboxes(
                image, pii_bboxes, show_text_annotation, use_greyscale_cmap
            )
            if display_image
            else None
        )

        return verify_image, ocr_bboxes, analyzer_results

    def eval_dicom_instance(
        self,
        instance: pydicom.dataset.FileDataset,
        ground_truth: List[dict],
        padding_width: int = 25,
        tolerance: int = 50,
        display_image: bool = False,
        use_metadata: bool = True,
        ocr_kwargs: Optional[dict] = None,
        ad_hoc_recognizers: Optional[List[PatternRecognizer]] = None,
        **text_analyzer_kwargs,
    ) -> Tuple[Optional[Image.Image], dict]:
        """Evaluate performance for a single DICOM instance.

        :param instance: Loaded DICOM instance including pixel data and metadata.
        :param ground_truth: Dictionary containing ground truth labels for the instance.
        :param padding_width: Padding width to use when running OCR.
        :param tolerance: Pixel distance tolerance for matching to ground truth.
        :param display_image: If the verification image is displayed and returned.
        :param use_metadata: Whether to redact text in the image that
        are present in the metadata.
        :param ocr_kwargs: Additional params for OCR methods.
        :param ad_hoc_recognizers: List of PatternRecognizer objects to use
        for ad-hoc recognizer.
        :param text_analyzer_kwargs: Additional values for the analyze method
        in ImageAnalyzerEngine.

        :return: Evaluation comparing redactor engine results vs ground truth.
        """
        # Verify detected PHI
        verify_image, ocr_results, analyzer_results = self.verify_dicom_instance(
            instance,
            padding_width,
            display_image,
            use_metadata,
            ocr_kwargs=ocr_kwargs,
            ad_hoc_recognizers=ad_hoc_recognizers,
            **text_analyzer_kwargs,
        )

        # Remove duplicate entities in results
        analyzer_results = self._remove_duplicate_entities(analyzer_results)

        # Get correct PHI text (all TP and FP)
        all_pos = self._label_all_positives(
            ground_truth, ocr_results, analyzer_results, tolerance
        )

        # Calculate evaluation metrics
        precision = self.calculate_precision(ground_truth, all_pos)
        recall = self.calculate_recall(ground_truth, all_pos)

        eval_results = {
            "all_positives": all_pos,
            "ground_truth": ground_truth,
            "precision": precision,
            "recall": recall,
        }

        return verify_image, eval_results

    @staticmethod
    def _remove_duplicate_entities(
        results: List[Dict], dup_pix_tolerance: int = 5
    ) -> List[Dict]:
        """Handle when a word is detected multiple times as different types of entities.

        :param results: List of detected PHI with bbox info.
        :param dup_pix_tolerance: Pixel difference tolerance for identifying duplicates.
        :return: Detected PHI with no duplicate entities.
        """
        dups = []
        sorted(results, key=lambda x: x.score, reverse=True)
        results_no_dups = []
        dims = ["left", "top", "width", "height"]

        # Check for duplicates
        for i in range(len(results) - 1):
            i_dims = {dim: results[i][dim] for dim in dims}

            # Ignore if we've already detected this dup combination
            for other in range(i + 1, len(results)):
                if i not in results_no_dups:
                    other_dims = {dim: results[other][dim] for dim in dims}
                    matching_dims = {
                        dim: abs(i_dims[dim] - other_dims[dim]) <= dup_pix_tolerance
                        for dim in dims
                    }
                    matching = list(matching_dims.values())

                    if all(matching):
                        lower_scored_index = (
                            other if results[other]["score"] < results[i]["score"] else i
                        )
                        dups.append(lower_scored_index)

        # Remove duplicates
        for i in range(len(results)):
            if i not in dups:
                results_no_dups.append(results[i])

        return results_no_dups

    def _label_all_positives(
        self,
        gt_labels_dict: List[dict],
        ocr_results: List[dict],
        detected_phi: List[ImageRecognizerResult],
        tolerance: int = 50,
    ) -> List[ImageRecognizerResult]:
        """Label all entities detected as PHI by using ground truth and OCR results.

        All positives (detected_phi) do not contain PHI labels and are thus
        difficult to work with intuitively. This method maps back to the
        actual PHI to each detected sensitive entity.

        :param gt_labels_dict: A list of Dictionaries with ground truth labels for a
        single DICOM instance.
        :param ocr_results: All detected text.
        :param detected_phi: Formatted analyzer_results.
        :param tolerance: Tolerance for exact coordinates and size data.
        :return: List of all positives, labeled.
        """
        all_pos = []

        # Cycle through each positive (TP or FP)
        for analyzer_result in detected_phi:
            # See if there are any ground truth matches
            all_pos, gt_match_found = self.bbox_processor.match_with_source(
                all_pos, gt_labels_dict, analyzer_result, tolerance
            )

            # If not, check back with OCR
            if not gt_match_found:
                all_pos, _ = self.bbox_processor.match_with_source(
                    all_pos, ocr_results, analyzer_result, tolerance
                )

        all_pos_results = self._remove_duplicate_entities(all_pos)

        return all_pos_results

    @staticmethod
    def __dict_to_image_recognizer_result(
        all_pos: List[Dict],
    ) -> List[ImageRecognizerResult]:
        fields = (
            "score",
            "left",
            "top",
            "width",
            "height",
        )
        to_recognizer_results = [
            {key: d[key] for key in fields if key in d} for d in all_pos
        ]

        all_pos_results = [
            ImageRecognizerResult(**pos) for pos in to_recognizer_results
        ]
        return all_pos_results

    @staticmethod
    def calculate_precision(
        gt: List[dict], all_pos: List[ImageRecognizerResult]
    ) -> float:
        """Calculate precision.

        :param gt: List of ground truth labels.
        :param all_pos: All Detected PHI (mapped back to have actual PHI text).
        :return: Precision value.
        """
        # Find True Positive (TP) and precision
        tp = [i for i in all_pos if i in gt]
        try:
            precision = len(tp) / len(all_pos)
        except ZeroDivisionError:
            precision = 0

        return precision

    @staticmethod
    def calculate_recall(gt: List[dict], all_pos: List[ImageRecognizerResult]) -> float:
        """Calculate recall.

        :param gt: List of ground truth labels.
        :param all_pos: All Detected PHI (mapped back to have actual PHI text).
        :return: Recall value.
        """
        # Find True Positive (TP) and precision
        tp = [i for i in all_pos if i in gt]
        try:
            recall = len(tp) / len(gt)
        except ZeroDivisionError:
            recall = 0

        return recall

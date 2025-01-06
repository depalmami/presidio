"""Integration test for dicom_image_pii_verify_engine.

Note we are not checking exact pixel data for the returned image
because that is covered by testing of the "verify" function in
the original parent ImagePiiVerifyEngine class.
"""

from PIL import Image
import pydicom

from presidio_image_redactor import DicomImagePiiVerifyEngine

PADDING_WIDTH = 25


def test_verify_correctly(
    get_mock_dicom_instance: pydicom.dataset.FileDataset,
    get_mock_dicom_verify_results: dict,
):
    """Test the verify_dicom_instance function.

    Args:
        get_mock_dicom_instance (pydicom.dataset.FileDataset): Loaded DICOM.
        get_mock_dicom_verify_results (dict): Dictionary with loaded results.
    """
    # Assign
    expected_ocr_results_labels = []
    for item in get_mock_dicom_verify_results["ocr_results_formatted"]:
        expected_ocr_results_labels.append(item["label"])

    # Act
    test_image_verify, test_ocr_results_formatted, _ = DicomImagePiiVerifyEngine().verify_dicom_instance(
        instance=get_mock_dicom_instance, padding_width=PADDING_WIDTH, display_image=True, ocr_kwargs=None
    )

    # Check most OCR results (labels) are the same
    # Don't worry about position since that is implied in analyzer results
    test_ocr_results_labels = []
    for item in test_ocr_results_formatted:
        test_ocr_results_labels.append(item["label"])
    test_common_labels = set(expected_ocr_results_labels).intersection(set(test_ocr_results_labels))
    test_all_labels = set(expected_ocr_results_labels).union(set(test_ocr_results_labels))

    # Assert
    assert isinstance(test_image_verify, Image.Image)
    assert len(test_common_labels) / len(test_all_labels) >= 0.5


def test_eval_dicom_instance(
    get_mock_dicom_instance: pydicom.dataset.FileDataset,
    get_mock_dicom_verify_results: dict,
):

    ground_truth = get_mock_dicom_verify_results["ground_truth"]
    verify_engine = DicomImagePiiVerifyEngine()
    _, test_eval_results = verify_engine.eval_dicom_instance(instance=get_mock_dicom_instance,
                                                             ground_truth=ground_truth)

    assert test_eval_results["precision"] == 1.0
    assert test_eval_results["recall"] == 1.0

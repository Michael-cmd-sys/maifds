"""
Unit tests for report models
"""

import pytest
from pydantic import ValidationError
from datetime import datetime

from src.models.report_model import Report, ReportMetadata


class TestReportMetadata:
    """Test ReportMetadata model"""

    def test_valid_metadata(self):
        """Test valid metadata creation"""
        metadata = ReportMetadata(
            platform="mobile", location="New York", device_info="iPhone 12"
        )
        assert metadata.platform == "mobile"
        assert metadata.location == "New York"

    def test_invalid_platform(self):
        """Test invalid platform raises error"""
        with pytest.raises(ValidationError):
            ReportMetadata(platform="invalid_platform")

    def test_optional_fields(self):
        """Test metadata with only required fields"""
        metadata = ReportMetadata(platform="web")
        assert metadata.platform == "web"
        assert metadata.location is None


class TestReport:
    """Test Report model"""

    def test_valid_report(self):
        """Test valid report creation"""
        report = Report(
            reporter_id="user_001",
            merchant_id="merchant_abc",
            report_type="fraud",
            rating=1,
            title="Test Report",
            description="This is a test report with sufficient length for validation.",
        )
        assert report.reporter_id == "user_001"
        assert report.merchant_id == "merchant_abc"
        assert report.report_type == "fraud"
        assert report.rating == 1
        assert isinstance(report.timestamp, datetime)
        assert len(report.report_id) > 0

    def test_text_cleaning(self):
        """Test text field cleaning"""
        report = Report(
            reporter_id="user_001",
            merchant_id="merchant_abc",
            report_type="fraud",
            title="  Test   Report  ",
            description="  This   is   a    test   report   with   extra   spaces.  ",
        )
        assert report.title == "Test Report"
        assert "   " not in report.description

    def test_missing_required_fields(self):
        """Test that missing required fields raise error"""
        with pytest.raises(ValidationError):
            Report(
                reporter_id="user_001",
                # Missing merchant_id
                report_type="fraud",
                title="Test",
                description="Test description",
            )

    def test_invalid_rating(self):
        """Test invalid rating values"""
        with pytest.raises(ValidationError):
            Report(
                reporter_id="user_001",
                merchant_id="merchant_abc",
                report_type="fraud",
                rating=10,  # Invalid: must be 1-5
                title="Test Report",
                description="This is a test report.",
            )

    def test_title_too_short(self):
        """Test title minimum length validation"""
        with pytest.raises(ValidationError):
            Report(
                reporter_id="user_001",
                merchant_id="merchant_abc",
                report_type="fraud",
                title="Hi",  # Too short
                description="This is a test report with sufficient length.",
            )

    def test_description_too_short(self):
        """Test description minimum length validation"""
        with pytest.raises(ValidationError):
            Report(
                reporter_id="user_001",
                merchant_id="merchant_abc",
                report_type="fraud",
                title="Test Report",
                description="Short",  # Too short
            )

    def test_invalid_id_format(self):
        """Test ID format validation"""
        with pytest.raises(ValidationError):
            Report(
                reporter_id="user@001",  # Invalid character
                merchant_id="merchant_abc",
                report_type="fraud",
                title="Test Report",
                description="This is a test report.",
            )

    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection"""
        with pytest.raises(ValidationError):
            Report(
                reporter_id="user_001",
                merchant_id="merchant_abc",
                report_type="fraud",
                title="Test Report",
                description="This is bad'; DROP TABLE reports; --",
            )

    def test_xss_detection(self):
        """Test XSS pattern detection"""
        with pytest.raises(ValidationError):
            Report(
                reporter_id="user_001",
                merchant_id="merchant_abc",
                report_type="fraud",
                title="Test Report",
                description="This is bad <script>alert('xss')</script>",
            )

    def test_to_dict_conversion(self):
        """Test converting report to dictionary"""
        report = Report(
            reporter_id="user_001",
            merchant_id="merchant_abc",
            report_type="fraud",
            title="Test Report",
            description="This is a test report.",
            metadata=ReportMetadata(platform="mobile"),
        )
        data = report.to_dict()

        assert data["reporter_id"] == "user_001"
        assert data["merchant_id"] == "merchant_abc"
        assert isinstance(data["timestamp"], str)
        assert "metadata_json" in data

    def test_from_dict_conversion(self):
        """Test creating report from dictionary"""
        data = {
            "report_id": "test-id-123",
            "timestamp": "2025-12-01T10:30:00",
            "reporter_id": "user_001",
            "merchant_id": "merchant_abc",
            "report_type": "fraud",
            "rating": 1,
            "title": "Test Report",
            "description": "This is a test report.",
            "transaction_id": None,
            "amount": None,
            "metadata_json": '{"platform": "mobile", "location": null, "device_info": null}',
        }

        report = Report.from_dict(data)
        assert report.reporter_id == "user_001"
        assert isinstance(report.timestamp, datetime)
        assert report.metadata.platform == "mobile"

    def test_optional_fields(self):
        """Test report with only required fields"""
        report = Report(
            reporter_id="user_001",
            merchant_id="merchant_abc",
            report_type="fraud",
            title="Test Report",
            description="This is a test report.",
        )
        assert report.rating is None
        assert report.transaction_id is None
        assert report.amount is None
        assert report.metadata is None

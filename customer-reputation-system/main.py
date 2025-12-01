"""
Demo script for testing the report ingestion system
"""

from src.ingestion.report_handler import ReportHandler
from config.logging_config import setup_logger

logger = setup_logger(__name__)


def create_sample_reports():
    """Create sample reports for testing"""
    return [
        {
            "reporter_id": "user_001",
            "merchant_id": "merchant_abc",
            "report_type": "fraud",
            "rating": 1,
            "title": "Unauthorized charge on my account",
            "description": "I noticed a charge of $150 that I did not authorize. The merchant charged me twice for the same transaction.",
            "transaction_id": "txn_12345",
            "amount": 150.00,
            "metadata": {
                "platform": "mobile",
                "location": "New York, USA",
                "device_info": "iPhone 12",
            },
        },
        {
            "reporter_id": "user_002",
            "merchant_id": "merchant_abc",
            "report_type": "service_issue",
            "rating": 2,
            "title": "Poor customer service experience",
            "description": "The merchant was unresponsive to my inquiries about delivery status. It took over 3 weeks to receive my order.",
            "transaction_id": "txn_67890",
            "amount": 75.50,
            "metadata": {"platform": "web", "location": "Los Angeles, USA"},
        },
        {
            "reporter_id": "user_003",
            "merchant_id": "merchant_xyz",
            "report_type": "fraud",
            "rating": 1,
            "title": "Fake product received",
            "description": "Ordered an authentic product but received a counterfeit item. The quality is terrible and does not match description.",
            "metadata": {"platform": "mobile", "location": "Chicago, USA"},
        },
        {
            "reporter_id": "user_001",
            "merchant_id": "merchant_xyz",
            "report_type": "technical",
            "rating": 3,
            "title": "Payment processing error",
            "description": "Encountered multiple errors while trying to complete payment. Had to try 5 times before it went through.",
            "transaction_id": "txn_11111",
            "amount": 200.00,
            "metadata": {"platform": "web"},
        },
        {
            "reporter_id": "user_004",
            "merchant_id": "merchant_def",
            "report_type": "service_issue",
            "rating": 4,
            "title": "Delayed shipping but good product",
            "description": "The product quality is excellent, but shipping took much longer than expected. Communication could be better.",
            "transaction_id": "txn_22222",
            "amount": 99.99,
            "metadata": {"platform": "mobile", "location": "Miami, USA"},
        },
    ]


def main():
    """Main demo function"""
    print("=" * 60)
    print("Customer Reputation System - Report Ingestion Demo")
    print("=" * 60)
    print()

    # Initialize handler
    handler = ReportHandler()

    # Get initial statistics
    print("Initial Statistics:")
    stats = handler.get_statistics()
    print(f"  Total Reports: {stats.get('total_reports', 0)}")
    print(f"  Total Reporters: {stats.get('total_reporters', 0)}")
    print(f"  Total Merchants: {stats.get('total_merchants', 0)}")
    print()

    # Submit sample reports
    print("Submitting sample reports...")
    print("-" * 60)

    sample_reports = create_sample_reports()
    submitted_ids = []

    for i, report_data in enumerate(sample_reports, 1):
        print(
            f"\n{i}. Submitting report from {report_data['reporter_id']} about {report_data['merchant_id']}"
        )
        result = handler.submit_report(report_data)

        if result["status"] == "success":
            print(f"   ✓ Success! Report ID: {result['report_id']}")
            submitted_ids.append(result["report_id"])
        else:
            print(f"   ✗ Failed: {result['message']}")

    print()
    print("-" * 60)

    # Get updated statistics
    print("\nUpdated Statistics:")
    stats = handler.get_statistics()
    print(f"  Total Reports: {stats.get('total_reports', 0)}")
    print(f"  Total Reporters: {stats.get('total_reporters', 0)}")
    print(f"  Total Merchants: {stats.get('total_merchants', 0)}")
    print()

    # Retrieve and display a sample report
    if submitted_ids:
        print("-" * 60)
        print("\nRetrieving first submitted report...")
        report = handler.get_report(submitted_ids[0])
        if report:
            print(f"\nReport Details:")
            print(f"  ID: {report.report_id}")
            print(f"  Reporter: {report.reporter_id}")
            print(f"  Merchant: {report.merchant_id}")
            print(f"  Type: {report.report_type}")
            print(f"  Rating: {report.rating}")
            print(f"  Title: {report.title}")
            print(f"  Description: {report.description[:100]}...")
            print(f"  Timestamp: {report.timestamp}")

    # Get reports by merchant
    print()
    print("-" * 60)
    print("\nReports for merchant_abc:")
    merchant_reports = handler.get_merchant_reports("merchant_abc")
    print(f"  Found {len(merchant_reports)} report(s)")
    for report in merchant_reports:
        print(
            f"  - {report.title} (Rating: {report.rating}, Type: {report.report_type})"
        )

    # Get reports by reporter
    print()
    print("-" * 60)
    print("\nReports by user_001:")
    reporter_reports = handler.get_reporter_reports("user_001")
    print(f"  Found {len(reporter_reports)} report(s)")
    for report in reporter_reports:
        print(f"  - {report.title} (Merchant: {report.merchant_id})")

    print()
    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Database configuration settings."""

class DatabaseSettings:
    """Database connection and configuration settings."""
    
    def __init__(self):
        # Connection
        self.url = "sqlite:///data/reports.db"
        self.pool_size = 5
        self.max_overflow = 10
        self.echo = False
        
        # Timeouts
        self.connection_timeout = 30
        self.query_timeout = 60
        
        # Migration
        self.auto_migrate = True
        self.migration_dir = "migrations"
        
        # Backup
        self.backup_enabled = True
        self.backup_interval = 3600  # seconds
        self.backup_dir = "data/backups"


# Global database settings instance
database_settings = DatabaseSettings()
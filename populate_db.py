"""
Auto-Create Database Setup Script for Credit Card Fraud Detection System
Creates database if it doesn't exist, then creates balanced dataset
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BalancedDatabaseSetup:
    def __init__(self):
        # Database configuration
        self.db_name = "Neobank_UNI"
        self.db_config = {
            "user": "postgres", 
            "password": os.environ.get("PASSWORD", "931579"),
            "host": "localhost",
            "port": "5432"
        }
        
        self.connection = None
        self.cursor = None
    
    def create_database_if_not_exists(self):
        """Create database if it doesn't exist"""
        try:
            # Connect to PostgreSQL server (not to specific database)
            conn = psycopg2.connect(**self.db_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (self.db_name,))
            exists = cursor.fetchone()
            
            if not exists:
                logger.info(f"🔨 Creating database '{self.db_name}'...")
                cursor.execute(f'CREATE DATABASE "{self.db_name}"')
                logger.info(f"✅ Database '{self.db_name}' created successfully")
            else:
                logger.info(f"✅ Database '{self.db_name}' already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating database: {e}")
            return False
    
    def connect(self):
        """Establish database connection"""
        try:
            # First ensure the database exists
            if not self.create_database_if_not_exists():
                return False
            
            # Now connect to the specific database
            db_config_with_db = self.db_config.copy()
            db_config_with_db["dbname"] = self.db_name
            
            self.connection = psycopg2.connect(**db_config_with_db)
            self.cursor = self.connection.cursor()
            logger.info(f"✅ Connected to PostgreSQL database '{self.db_name}'")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            logger.info("💡 Make sure PostgreSQL is running and credentials are correct")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("🔌 Disconnected from database")
    
    def drop_existing_table(self):
        """Drop existing table if it exists"""
        try:
            self.cursor.execute("DROP TABLE IF EXISTS credit_card_transactions CASCADE;")
            self.connection.commit()
            logger.info("🗑️ Dropped existing credit_card_transactions table")
        except Exception as e:
            logger.error(f"❌ Error dropping table: {e}")
            self.connection.rollback()
    
    def create_credit_card_table(self):
        """Create the credit_card_transactions table"""
        create_table_query = """
        CREATE TABLE credit_card_transactions (
            time_id SERIAL PRIMARY KEY,
            v1 NUMERIC, v2 NUMERIC, v3 NUMERIC, v4 NUMERIC, v5 NUMERIC,
            v6 NUMERIC, v7 NUMERIC, v8 NUMERIC, v9 NUMERIC, v10 NUMERIC,
            v11 NUMERIC, v12 NUMERIC, v13 NUMERIC, v14 NUMERIC, v15 NUMERIC,
            v16 NUMERIC, v17 NUMERIC, v18 NUMERIC, v19 NUMERIC, v20 NUMERIC,
            v21 NUMERIC, v22 NUMERIC, v23 NUMERIC, v24 NUMERIC, v25 NUMERIC,
            v26 NUMERIC, v27 NUMERIC, v28 NUMERIC,
            amount NUMERIC NOT NULL,
            class INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_time_id ON credit_card_transactions(time_id);
        CREATE INDEX idx_class ON credit_card_transactions(class);
        CREATE INDEX idx_amount ON credit_card_transactions(amount);
        """
        
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Created credit_card_transactions table")
            return True
        except Exception as e:
            logger.error(f"❌ Error creating table: {e}")
            self.connection.rollback()
            return False
    
    def insert_balanced_data(self, records_per_class=500):
        """Insert equal number of class 0 and class 1 records"""
        logger.info(f"📝 Inserting {records_per_class} records for each class (0 and 1)...")
        
        try:
            np.random.seed(42)  # For reproducibility
            
            # Insert query
            insert_query = """
            INSERT INTO credit_card_transactions 
            (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
             v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount, class)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Generate and insert class 0 records
            logger.info(f"📊 Inserting {records_per_class} records with class = 0")
            for i in range(records_per_class):
                # Generate random V1-V28 features
                v_features = np.random.uniform(-3, 3, 28).tolist()
                # Generate random amount
                amount = np.random.uniform(1, 1000)
                
                record = v_features + [amount, 0]  # class = 0
                self.cursor.execute(insert_query, record)
                
                if (i + 1) % 100 == 0:
                    self.connection.commit()
                    logger.info(f"   ✅ Inserted {i + 1}/{records_per_class} class 0 records")
            
            self.connection.commit()
            
            # Generate and insert class 1 records
            logger.info(f"📊 Inserting {records_per_class} records with class = 1")
            for i in range(records_per_class):
                # Generate random V1-V28 features
                v_features = np.random.uniform(-3, 3, 28).tolist()
                # Generate random amount
                amount = np.random.uniform(1, 1000)
                
                record = v_features + [amount, 1]  # class = 1
                self.cursor.execute(insert_query, record)
                
                if (i + 1) % 100 == 0:
                    self.connection.commit()
                    logger.info(f"   ✅ Inserted {i + 1}/{records_per_class} class 1 records")
            
            self.connection.commit()
            
            # Verify the data
            self.cursor.execute("SELECT class, COUNT(*) FROM credit_card_transactions GROUP BY class ORDER BY class")
            results = self.cursor.fetchall()
            
            logger.info("🎉 Successfully inserted balanced data:")
            for class_val, count in results:
                logger.info(f"   Class {class_val}: {count} records")
            
            total_records = records_per_class * 2
            logger.info(f"📊 Total records: {total_records}")
            logger.info(f"⚖️ Balance: 50% class 0, 50% class 1")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inserting data: {e}")
            self.connection.rollback()
            return False
    
    def verify_balance(self):
        """Verify the class balance in the database"""
        try:
            self.cursor.execute("""
                SELECT 
                    class,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                FROM credit_card_transactions 
                GROUP BY class 
                ORDER BY class
            """)
            
            results = self.cursor.fetchall()
            
            logger.info("📊 Current class distribution:")
            for class_val, count, percentage in results:
                logger.info(f"   Class {class_val}: {count} records ({percentage}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying balance: {e}")
            return False
    
    def check_table_exists(self):
        """Check if the table exists and show its structure"""
        try:
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'credit_card_transactions'
                );
            """)
            exists = self.cursor.fetchone()[0]
            
            if exists:
                logger.info("✅ Table 'credit_card_transactions' exists")
                
                # Show table structure
                self.cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = 'credit_card_transactions' 
                    ORDER BY ordinal_position;
                """)
                
                columns = self.cursor.fetchall()
                logger.info("📋 Table structure:")
                for col_name, col_type, nullable in columns:
                    logger.info(f"   - {col_name}: {col_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
                
                # Show record count
                self.cursor.execute("SELECT COUNT(*) FROM credit_card_transactions;")
                count = self.cursor.fetchone()[0]
                logger.info(f"📊 Total records in table: {count}")
                
                if count > 0:
                    self.verify_balance()
            else:
                logger.info("❌ Table 'credit_card_transactions' does not exist")
            
            return exists
            
        except Exception as e:
            logger.error(f"❌ Error checking table: {e}")
            return False

def main():
    """Main setup function"""
    setup = BalancedDatabaseSetup()
    
    try:
        print("🚀 STARTING DATABASE SETUP...")
        print("This script will:")
        print("1. Create database if it doesn't exist")
        print("2. Create table with balanced data (equal class 0 and 1)")
        
        # Connect to database (will create if needed)
        if not setup.connect():
            logger.error("❌ Failed to connect to database. Exiting.")
            return
        
        # Check if table already exists
        table_exists = setup.check_table_exists()
        
        if table_exists:
            recreate = input("\n🔄 Table already exists. Recreate with new data? (y/N): ").strip().lower()
            if recreate != 'y':
                logger.info("✅ Keeping existing table")
                return
        
        print("\n🔧 BALANCED DATABASE SETUP:")
        records_per_class = input("Enter number of records per class (default 500): ").strip()
        records_per_class = int(records_per_class) if records_per_class else 500
        
        confirm = input(f"\nThis will create {records_per_class * 2} total records. Continue? (y/N): ").strip().lower()
        
        if confirm != 'y':
            logger.info("❌ Setup cancelled by user")
            return
        
        # Drop and recreate table
        setup.drop_existing_table()
        if setup.create_credit_card_table():
            if setup.insert_balanced_data(records_per_class):
                logger.info("🎉 SUCCESS! Database setup completed successfully")
            else:
                logger.error("❌ Failed to insert data")
        else:
            logger.error("❌ Failed to create table")
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Setup interrupted by user")
    except ValueError as e:
        logger.error(f"❌ Invalid input: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        setup.disconnect()
        
    print("\n✅ Database setup complete!")
    print("You can now run your fraud detection server with: python main.py")

if __name__ == "__main__":
    main()
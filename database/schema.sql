CREATE DATABASE IF NOT EXISTS disease_db;
USE disease_db;

-- Users (login/register + role)
CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(100) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) NOT NULL DEFAULT 'user', -- 'admin' hoặc 'user'
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions history (dashboard + table)
CREATE TABLE IF NOT EXISTS predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  pregnancies INT,
  glucose FLOAT,
  blood_pressure FLOAT,
  bmi FLOAT,
  age INT,
  result VARCHAR(10),
  probability FLOAT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_result ON predictions(result);


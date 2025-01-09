SELECT response_status_code,count(*)
FROM image_processing
GROUP BY response_status_code
ORDER BY response_status_code ASC;
-- -- SQLite
-- SELECT id, pdf_file, timestamp, image, image_path, success_status, response_status_code, response, error_message, embedding
-- FROM image_processing
-- WHERE embedding is NOT NULL;

-- CREATE TABLE image_processing_backup AS SELECT * FROM image_processing;

-- -- Create new table with the embedding column
-- CREATE TABLE image_processing_new (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     pdf_file TEXT,
--     timestamp TEXT,
--     image TEXT,
--     image_path TEXT,
--     success_status BOOLEAN,
--     response_status_code INTEGER,
--     response TEXT,
--     error_message TEXT,
--     embedding BLOB
-- );

-- -- Copy data from old table to new table
-- INSERT INTO image_processing_new (id, pdf_file, timestamp, image, image_path, success_status, response_status_code, response, error_message)
-- SELECT id, pdf_file, timestamp, image, image_path, success_status, response_status_code, response, error_message 
-- FROM image_processing_backup;

-- -- Drop the old table
-- DROP TABLE image_processing;

-- -- Rename the new table
-- ALTER TABLE image_processing_new RENAME TO image_processing;
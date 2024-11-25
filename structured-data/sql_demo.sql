-- Relational Database Schema
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    location VARCHAR(100)
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2)
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    price_per_unit DECIMAL(10,2),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Sample data
INSERT INTO customers VALUES 
(1, 'John Doe', 'john@email.com', 'New York'),
(2, 'Jane Smith', 'jane@email.com', 'Los Angeles');

INSERT INTO products VALUES
(1, 'Laptop', 'Electronics', 999.99),
(2, 'Headphones', 'Electronics', 99.99);

INSERT INTO orders VALUES
(1, 1, '2024-01-01', 1099.98),
(2, 2, '2024-01-02', 99.99);

INSERT INTO order_items VALUES
(1, 1, 1, 999.99),
(1, 2, 1, 99.99),
(2, 2, 1, 99.99);
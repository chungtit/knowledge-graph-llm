-- Cypher queries to convert to Knowledge Graph
-- Note: These would be executed in a Neo4j environment

// Create Customer nodes
CREATE (:Customer {
    customerId: 1,
    name: 'John Doe',
    email: 'john@email.com',
    location: 'New York'
});

// Create Product nodes
CREATE (:Product {
    productId: 1,
    name: 'Laptop',
    category: 'Electronics',
    price: 999.99
});

// Create Order nodes and relationships
MATCH (c:Customer {customerId: 1})
CREATE (o:Order {
    orderId: 1,
    orderDate: '2024-01-01',
    totalAmount: 1099.98
})
CREATE (c)-[:PLACED]->(o);

// Create OrderItem relationships
MATCH (o:Order {orderId: 1})
MATCH (p:Product {productId: 1})
CREATE (o)-[:CONTAINS {
    quantity: 1,
    pricePerUnit: 999.99
}]->(p);
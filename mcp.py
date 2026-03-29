import asyncio
# from mcp.server.fastmcp import FastMCP
# from transactional_db import PRODUCTS_TABLE

mcp = FastMCP("ecommerce_tools")

@mcp.tool()
async def check_inventory(product_name: str) -> str:
    """Search inventory for a product by product name."""
    await asyncio.sleep(1)
    matches = []
    for sku, product in PRODUCTS_TABLE.items():
        if product_name.lower() in product["name"].lower():
            matches.append(
                f"{product['name']} (SKU: {sku}) — Stock: {product['stock']}"
            )
    return "\n".join(matches) if matches else "No matching products found."

if __name__ == "__main__":
    mcp.run(transport="stdio")
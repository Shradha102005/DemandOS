from io import BytesIO
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_forecast():
    r = client.get('/api/forecast/SKU-001?horizon=7')
    assert r.status_code == 200 and len(r.json()['mean']) == 7

def test_upload_and_dynamic_forecast():
    csv = b'date,sku,demand\n2025-01-01,A-1,10\n2025-01-02,A-1,12\n2025-01-03,A-1,11\n'
    r = client.post('/api/dataset/upload', files={'file': ('sales.csv', BytesIO(csv), 'text/csv')})
    assert r.status_code == 200 and r.json()['summary']['sku_count'] == 1
    assert client.get('/api/forecast/A-1?horizon=7').status_code == 200
    client.post('/api/dataset/reset')

def test_sample_and_flexible_headers():
    csv = b'Date,Store ID,Product ID,Units Sold\n1/1/2022,S001,P0001,127\n1/1/2022,S001,P0002,150\n'
    r = client.post('/api/dataset/upload', files={'file': ('retail.csv', BytesIO(csv), 'text/csv')})
    assert r.status_code == 200 and r.json()['summary']['sku_count'] == 2
    assert client.get('/api/forecast/P0001?horizon=7').status_code == 200
    client.post('/api/dataset/reset')

def test_inventory_and_whatif():
    assert len(client.get('/api/inventory/health').json()) == 5
    r = client.post('/api/whatif', json={'sku':'SKU-002','discount_pct':20,'promo':True})
    assert r.status_code == 200 and r.json()['revenue_delta'] > 0

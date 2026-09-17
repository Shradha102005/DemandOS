from io import BytesIO
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_sample_dataset_headers_and_tabs():
    content = b"Date\tStore ID\tProduct ID\tCategory\tRegion\tInventory Level\tUnits Sold\tUnits Ordered\tDemand Forecast\tPrice\tDiscount\tWeather Condition\tHoliday/Promotion\tCompetitor Pricing\tSeasonality\n1/1/2022\tS001\tP0001\tGroceries\tNorth\t231\t127\t55\t135.47\t33.5\t20\tRainy\t0\t29.69\tAutumn\n1/1/2022\tS001\tP0002\tToys\tSouth\t204\t150\t66\t144.04\t63.01\t20\tSunny\t0\t66.16\tAutumn\n"
    response = client.post('/api/dataset/upload', files={'file': ('sample.csv', BytesIO(content), 'text/csv')})
    assert response.status_code == 200
    assert response.json()['summary']['sku_count'] == 2
    assert client.get('/api/forecast/P0001?horizon=7').status_code == 200
    client.post('/api/dataset/reset')

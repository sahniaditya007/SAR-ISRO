from azureml.core import Workspace

ws = Workspace.create(name='your_workspace_name',
                      subscription_id='your_subscription_id',
                      resource_group='your_resource_group',
                      location='your_location')
ws.write_config()

from azureml.core import Workspace

ws = Workspace.create(name='your_workspace_name',
                      subscription_id='your_subscription_id',
                      resource_group='your_resource_group',
                      location='your_location')
ws.write_config()

from pystac_client import Client
import planetary_computer as pc
import xarray as xr

stac_api_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
client = Client.open(stac_api_url)

search = client.search(
    collections=["sentinel-1-grd"],
    bbox=[longitude_min, latitude_min, longitude_max, latitude_max],
    datetime="2020-01-01/2020-12-31",
    limit=10
)

items = list(search.get_items())
item = items[0]
signed_asset = pc.sign(item.assets["vv"])

arr = xr.open_rasterio(signed_asset.href)


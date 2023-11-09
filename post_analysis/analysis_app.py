import streamlit as st
from metaflow import Flow
import geopandas as gpd
import pandas as pd
import pydeck as pdk

#Goal: 1 data frame of all model results
#       Select on that, we get a dataframe of specific results for that model
#       We also get: metrics map, residuals, various other plots
#       Side by side??

st.title('Analysis Test')

@st.cache_data
def load_results():
    return pd.read_csv('./data/temp/prediction_metrics.csv').astype({'item_id': 'string'})

@st.cache_data
def load_gdf() -> gpd.GeoDataFrame:
    munis =  gpd.read_file('./data/brazil/munis/munis_simple.shp').astype({'CD_MUN': 'string'}).to_crs('EPSG:4326')
    munis['CD_MUN'] = munis['CD_MUN'].str.slice(stop=-1)
    return munis.rename(columns={'CD_MUN': 'item_id'})

#@st.cache_resource
def join_items(gdf, results):
    return gdf.merge(results, on='item_id')
    


results = load_results()
gdf = load_gdf()

st.write(results)
#st.write(gdf)


joined = join_items(gdf, results)

#st.write(joined)

def update_deck():
    layer.get_elevation = option

    deck.update()
    pass

option = st.selectbox(
    'Metric',
    ('MSE', 'abs_error'),
    on_change=update_deck)

st.write(option)

layer = pdk.Layer(
           'GeoJsonLayer',
           data=joined,
           opacity=0.8,
    stroked=True,
    filled=True,
           #get_polygon = joined.geometry,
           #stroked=False,
    # processes the data as a flat longitude-latitude pair
    #get_polygon='-',
    get_fill_color='[180,0,180, abs_error+20]',
        #    radius=200,
        #    elevation_scale=4,
        #    elevation_range=[0, 1000],
           pickable=True,
           #get_elevation = 'abs_target_sum',
           get_elevation = 'abs_error',
           elevation_scale = 500,
           extruded=True,
        #    extruded=True,
        )
deck = pdk.Deck(
    map_style=None,
    layers=[layer],
    tooltip={
   "html": "<b>{NM_MUN}</b> <br><b>Abs Error:</b> {abs_error} <br> <b>Actual Case Rate:</b> {abs_target_sum}",
   "style": {
        "backgroundColor": "steelblue",
        "color": "white"
   }
},
    initial_view_state=pdk.ViewState(latitude=-14,
        longitude=-50,
        zoom=4,
        pitch=30)
)

st.pydeck_chart(deck)

#st.write(layer.get_elevation)
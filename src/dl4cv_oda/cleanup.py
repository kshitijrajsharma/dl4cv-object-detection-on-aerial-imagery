import geopandas as gpd
from shapely.geometry import box


def parse_tags(tags_str):
    result = {}
    for pair in tags_str.split(","):
        if pair:
            key, value = pair.split("=>")
            result[key.strip('"')] = value.strip('"')
    return result


def clean_osm_data(input_path, output_point_path, output_box_path, target=None):
    data = gpd.read_file(input_path)
    data = data.dropna(subset=["other_tags"])

    data["tags_dict"] = data["other_tags"].apply(parse_tags)
    data["natural"] = data["tags_dict"].apply(lambda x: x.get("natural"))
    data["species"] = data["tags_dict"].apply(lambda x: x.get("species"))

    mapping = {
        "Mangifera indica": "Mango",
        "Mangifera indica L.": "Mango",
        "Cocos nucifera": "Coconut",
        "Cocus nucifera": "Coconut",
        "cocos nucifera": "Coconut",
        "Cocos nucifera L.": "Coconut",
        "Cocos c": "Coconut",
        "Carica tree": "Papaya",
        "Carica papaya": "Papaya",
        "Musaceae": "Banana",
        "Musa": "Banana",
    }

    data["species_mapped"] = data["species"].map(mapping)
    data = data.dropna(subset=["species_mapped"])

    if target:
        data = data[data["species_mapped"] == target]

    data = data[["osm_id", "natural", "species", "species_mapped", "geometry"]]

    data.to_crs(epsg=3857, inplace=True)

    buffered = data.copy()
    buffered["lon"] = buffered.geometry.x
    buffered["lat"] = buffered.geometry.y
    buffered["geometry"] = buffered["geometry"].buffer(3)
    buffered["geometry"] = buffered.geometry.apply(lambda g: box(*g.bounds))

    data.to_file(output_point_path, driver="GeoJSON")
    buffered.to_file(output_box_path, driver="GeoJSON")

    return len(data)

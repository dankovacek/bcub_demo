Search.setIndex({"docnames": ["0_intro", "notebooks/1_Getting_Started", "notebooks/2_DEM_Preprocessing", "notebooks/3_Pour_Point_Extraction", "notebooks/4_Basin_Delineation", "notebooks/5_Attribute_Extraction", "notebooks/6_Postgres"], "filenames": ["0_intro.md", "notebooks/1_Getting_Started.ipynb", "notebooks/2_DEM_Preprocessing.ipynb", "notebooks/3_Pour_Point_Extraction.ipynb", "notebooks/4_Basin_Delineation.ipynb", "notebooks/5_Attribute_Extraction.ipynb", "notebooks/6_Postgres.ipynb"], "titles": ["BC Ungauged Basin Database (BCUB)", "1. Getting Started", "2. Digital Elevation Preprocessing", "3. Pour Point Extraction", "4. Basin Delineation", "5. Basin Attribute Extraction", "6. Basin Attribute Extraction - Postgres + PostGIS Method"], "terms": {"The": [0, 1, 2, 3, 5], "pour": [0, 2, 4, 5, 6], "point": [0, 1, 2, 4, 5, 6], "dataset": [0, 1, 2, 5, 6], "appear": [0, 3], "abov": [0, 1, 3, 5], "like": [0, 1, 2, 4], "black": [0, 2], "pointil": 0, "dot": 0, "while": [0, 2, 3], "exist": [0, 1, 2, 3, 4, 5], "histor": 0, "streamflow": 0, "monitor": [0, 5, 6], "station": 0, "ar": [0, 1, 2, 3, 4, 5, 6], "green": [0, 3], "british": [0, 1], "columbia": [0, 1], "intend": 0, "support": 0, "water": [0, 5], "resourc": [0, 3, 6], "research": 0, "name": [0, 1, 3], "optim": 0, "network": 0, "There": [0, 3], "sever": [0, 2, 5], "wai": 0, "us": [0, 1, 2, 3, 4, 5, 6], "inform": [0, 1, 3, 5, 6], "provid": [0, 1, 2, 6], "thi": [0, 1, 2, 3, 4, 5, 6], "repositori": 0, "A": [0, 3, 5], "minim": [0, 5], "contain": [0, 1, 6], "attribut": [0, 3], "nearli": 0, "1": [0, 2, 3, 4, 5], "million": [0, 2], "around": [0, 3], "file": [0, 2, 3, 5, 6], "doe": [0, 3], "polygon": [0, 1, 2, 3, 5, 6], "sinc": [0, 2, 3], "requir": [0, 1, 3], "veri": [0, 3], "larg": [0, 3, 4, 5, 6], "disk": [0, 4], "space": [0, 2, 4], "we": [0, 1, 2, 3, 4, 5, 6], "cannot": 0, "host": 0, "data": [0, 1, 2, 3, 4, 6], "an": [0, 1, 2, 3, 4], "expand": 0, "set": [0, 2, 3, 4, 5, 6], "compress": 0, "parquet": [0, 4], "plu": 0, "accompani": 0, "total": [0, 3, 4, 5], "approxim": [0, 5], "60": [0, 3, 4], "gb": [0, 4], "notebook": [0, 1, 2, 3, 5, 6], "demonstr": [0, 3], "complet": [0, 6], "process": [0, 1, 2, 3, 4, 5, 6], "gener": [0, 2, 3, 4, 6], "purpos": [0, 3], "extract": [0, 2], "carri": [0, 3, 4, 6], "out": [0, 2, 3, 4, 5, 6], "smaller": [0, 3, 4], "region": [0, 3, 4, 5, 6], "vancouv": [0, 1, 2, 3, 4, 5, 6], "island": [0, 1, 2, 3, 4, 5, 6], "how": [0, 3, 5], "do": [0, 2, 3, 5], "i": [0, 1, 2, 3, 5, 6], "other": [0, 3, 4, 6], "get": [0, 2, 4, 5], "error": [0, 1], "x": [0, 1, 2, 3], "fix": [0, 5], "you": [0, 1, 2, 3, 4, 5], "welcom": 0, "start": [0, 4], "discuss": [0, 1], "github": [0, 5], "repo": 0, "can": [0, 1, 2, 3, 4, 5, 6], "contribut": 0, "bring": 0, "up": [0, 1, 2, 3, 4, 6], "idea": [0, 3], "go": [0, 2, 5], "ahead": [0, 1], "make": [0, 2], "pull": [0, 1], "request": 0, "all": [0, 1, 3, 4, 5, 6], "code": [0, 2, 3], "creat": [0, 1, 2, 3, 4, 5, 6], "from": [0, 1, 2, 3, 4, 5, 6], "under": [0, 1, 3], "creativ": 0, "common": [0, 4], "4": [0, 3, 5], "0": [0, 1, 2, 3, 4, 5], "intern": 0, "In": [1, 2, 3, 5], "instal": 1, "python": 1, "librari": [1, 2], "usg": [1, 2, 5], "map": [1, 3, 4, 5], "applic": 1, "programmat": 1, "bcub": [1, 2, 3, 5, 6], "databas": [1, 5, 6], "wa": [1, 2, 3, 5], "develop": [1, 5], "program": 1, "languag": 1, "version": 1, "3": [1, 5], "10": [1, 3, 4, 5], "ha": [1, 2, 3], "number": [1, 2, 3, 4], "depenc": 1, "which": [1, 2, 3, 5, 6], "txt": 1, "ubuntu": 1, "linux": 1, "window": [1, 3], "mac": 1, "much": 1, "differ": 1, "howev": 1, "should": [1, 2, 5], "expect": [1, 5], "some": [1, 2, 3, 6], "run": [1, 2, 4, 5], "hardwar": [1, 2], "softwar": 1, "disagr": [1, 3], "upon": [1, 2, 6], "your": [1, 2, 6], "combin": [1, 3, 6], "os": [1, 2, 3, 4, 5], "ymmv": 1, "herein": 1, "packag": 1, "pip": 1, "togeth": 1, "varieti": 1, "sourc": [1, 5], "signific": 1, "effort": 1, "aquir": 1, "It": [1, 2, 3], "recommend": 1, "new": [1, 2, 3, 4], "virtual": [1, 2], "assum": [1, 5], "user": 1, "familiar": [1, 2], "conda": 1, "virtualenv": 1, "basic": 1, "first": [1, 4, 5], "step": [1, 2, 3, 5], "our": [1, 3], "basin": [1, 2, 3], "geometri": [1, 2, 4, 5], "repres": [1, 2, 3], "studi": 1, "area": [1, 2, 3, 4, 5], "retriev": [1, 3, 4], "digit": 1, "elev": [1, 3, 5], "section": 1, "3d": 1, "north": [1, 5, 6], "america": 1, "roughli": [1, 2, 5], "30m": 1, "resolut": [1, 2, 3, 4, 5], "ve": 1, "found": [1, 3, 6], "describ": [1, 2, 3, 5], "about": [1, 2], "32": 1, "000": 1, "km": [1, 2, 3, 4, 5], "2": [1, 3, 4, 5], "If": [1, 2, 4], "geograph": [1, 5], "cr": [1, 2, 3, 4, 5], "epsg": [1, 2, 3, 4, 5], "4269": 1, "4326": [1, 5], "look": [1, 2, 3], "bit": 1, "skew": 1, "imag": [1, 2, 3], "below": [1, 2, 3, 5, 6], "left": [1, 3], "project": [1, 2, 3, 5], "equal": [1, 5], "bc": [1, 2], "alber": [1, 2], "3005": [1, 2, 3, 4, 5], "take": [1, 2, 4, 5], "more": [1, 2, 3, 5], "spatial": [1, 3, 5, 6], "form": [1, 2, 3], "right": 1, "let": [1, 2, 3], "s": [1, 2, 3, 4, 5], "try": [1, 2, 5], "load": [1, 2, 3, 4, 5], "save": [1, 3, 5], "geojson": [1, 2, 3, 4, 5, 6], "format": [1, 3, 4, 5, 6], "region_polygon": [1, 2, 3, 5], "vancouver_island": [1, 2, 3, 4, 5], "import": [1, 2, 4, 5], "geopanda": [1, 2, 3, 4, 5], "gpd": [1, 2, 3, 4, 5], "base_dir": [1, 2, 3, 4, 5], "getcwd": [1, 2, 3, 4, 5], "polygon_path": [1, 3, 4, 5], "path": [1, 2, 3, 4, 5], "join": [1, 2, 3, 4, 5], "df": [1, 5], "read_fil": [1, 2, 3, 4, 5], "did": 1, "modulenotfound": 1, "so": [1, 2, 3, 5], "mean": [1, 5], "need": [1, 2, 3, 4, 5], "done": 1, "one": [1, 2, 3], "encount": 1, "e": [1, 2, 3, 4, 5], "Or": 1, "onc": [1, 6], "command": [1, 2, 5], "r": 1, "visit": 1, "tnm": 1, "see": [1, 2, 3], "product": [1, 2], "3dep": [1, 2, 5], "tab": 1, "hand": 1, "side": 1, "draw": 1, "click": 1, "search": 1, "yield": [1, 2, 5], "16": [1, 5], "correspond": [1, 3, 5], "intersect": [1, 3, 4, 5], "box": 1, "next": [1, 2, 3, 5], "collaps": 1, "option": 1, "csv": [1, 5], "link": [1, 5], "alreadi": [1, 3, 4, 5], "been": [1, 3, 4], "content": [1, 2, 3], "folder": [1, 3, 5], "now": [1, 2, 5], "panda": [1, 3, 4, 5], "pd": [1, 3, 4, 5], "dirnam": [1, 2, 3, 5], "gone": 1, "preview": 1, "tabl": [1, 3, 5, 6], "14th": 1, "column": [1, 3], "index": [1, 3, 5], "tif": [1, 2, 3, 4, 5], "links_path": 1, "download_link": 1, "read_csv": 1, "header": 1, "none": [1, 2, 3], "usecol": 1, "14": [1, 5], "where": [1, 2, 3], "local": [1, 2, 6], "save_path": 1, "dem_path": 1, "mkdir": [1, 3, 4, 5], "def": [1, 4, 5], "download_fil": 1, "specifi": [1, 5], "directori": 1, "filenam": 1, "split": [1, 2, 3, 4], "f": [1, 2, 3, 4, 5], "wget": 1, "p": [1, 4, 5], "out_path": 1, "print": [1, 2, 3, 4, 5], "system": [1, 2, 3, 5, 6], "_": [1, 2, 3, 4, 5], "row": [1, 3, 5], "iterrow": [1, 3, 5], "valu": [1, 2, 3, 4, 5], "rioxarrai": [1, 2], "its": [1, 2, 3], "properti": 1, "rxr": [1, 2], "dem_fil": [1, 2], "listdir": [1, 2, 4, 5], "test_fil": 1, "open_rasterio": [1, 2], "y": [1, 3], "coordin": [1, 3, 5], "decim": [1, 2, 5], "degre": [1, 2, 5], "follow": [1, 3], "rio": [1, 2, 3, 4, 5], "to_epsg": 1, "build": 1, "raster": [1, 2, 4, 5, 6], "vrt": [1, 2], "enabl": 1, "oper": [1, 4, 6], "mosaic": 1, "vrt_path": [1, 2], "usgs_3dep_mosaic_4269": [1, 2], "vrt_command": 1, "gdalbuildvrt": 1, "highest": 1, "a_sr": 1, "fail": [1, 5], "check": [1, 2, 3, 4, 5], "ensur": [1, 5], "have": [1, 2, 3, 4, 5], "For": [1, 3, 4], "maco": 1, "tutori": [1, 2], "mai": [1, 2, 3, 6], "level": [1, 5], "sudo": 1, "apt": 1, "libgdal": 1, "dev": 1, "ll": [2, 3, 4], "merg": [2, 3, 5], "tile": 2, "them": [2, 4], "whitebox": [2, 4], "ultim": 2, "delin": [2, 3, 6], "arcgi": 2, "qgi": 2, "those": 2, "tool": 2, "prefer": 2, "d8": [2, 3], "util": [2, 3, 4, 5], "dem_dir": [2, 5], "endswith": [2, 4], "open": [2, 3, 4, 5], "sampl": [2, 4, 5], "dem_fpath": [2, 5], "affin": [2, 3, 5], "retrieve_rast": [2, 3, 4, 5], "dem_resolut": 2, "dx": [2, 3, 5], "dy": [2, 3, 5], "int": [2, 3, 4], "ab": [2, 3, 4, 5], "m": [2, 3, 5], "don": [2, 3, 4], "t": [2, 3, 4], "end": [2, 5, 6], "comput": 2, "ocean": 2, "surround": 2, "identifi": [2, 3], "river": [2, 3], "drain": 2, "mask_path": 2, "output_dem_path": [2, 5], "vancouver_island_": 2, "mask": [2, 5], "gtype": 2, "geom_typ": [2, 3, 4], "is_valid": [2, 5], "valid": [2, 4], "shorelin": [2, 3], "function": [2, 3, 4, 5], "minut": [2, 5], "depend": [2, 6], "note": [2, 3, 5], "result": [2, 3], "1gb": 2, "variou": 2, "1x": 2, "ram": [2, 4, 6], "gdalwarp": [2, 5], "s_sr": [2, 5], "cutlin": [2, 5], "cl": [2, 5], "crop_to_cutlin": [2, 5], "multi": [2, 5], "gtiff": [2, 5], "wo": [2, 5], "num_thread": [2, 5], "all_cpu": [2, 5], "work": [2, 3, 4, 5, 6], "distanc": [2, 3, 5], "want": [2, 3, 5], "specif": [2, 3], "locat": [2, 3], "here": [2, 3, 4, 5], "reproject": [2, 3, 5], "pixel": [2, 3, 5], "low": 2, "re": [2, 5], "new_cr": 2, "dem_path_reproject": 2, "replac": [2, 4, 5], "lr": 2, "true": [2, 3, 4, 5], "default": [2, 3], "to_rast": 2, "lr_shape": 2, "shape": [2, 3, 4, 5], "n_pix": 2, "img": 2, "2e": 2, "els": [2, 3, 4, 5], "fname": 2, "skip": [2, 5], "plot": 2, "colour": 2, "time": [2, 3, 4, 5], "200": [2, 3], "feel": 2, "free": 2, "past": 2, "cell": [2, 3, 4, 5], "rasterio": 2, "show": [2, 3], "unprocess": 2, "pit": [2, 5], "depress": [2, 5], "prevent": [2, 3], "resolv": 2, "fill": [2, 5], "surfac": [2, 3], "base": 2, "each": [2, 3, 4, 5], "lowest": 2, "neighbour": 2, "pointer": [2, 3], "upstream": 2, "hillslop": 2, "threshold": [2, 3], "realiti": 2, "factor": 2, "vari": 2, "issu": [2, 5], "googl": 2, "colab": 2, "wbt": [2, 4], "whiteboxtool": [2, 4], "chang": [2, 3, 5, 6], "detail": [2, 3, 5], "log": [2, 5], "verbos": [2, 4], "fals": [2, 3, 4], "absolut": 2, "url": 2, "full": [2, 4], "home": [2, 3], "danbot": [2, 3], "document": [2, 3], "23": [2, 3], "oppos": 2, "rel": [2, 3], "filepath": 2, "filled_dem_path": 2, "_fill": 2, "fill_depress": 2, "fix_flat": 2, "flat_incr": 2, "max_depth": 2, "callback": [2, 4], "d8_pointer_path": 2, "vancouver_island_d8_point": 2, "d8_pointer": 2, "esri_pntr": [2, 4], "acc_path": [2, 3], "vancouver_island_acc": 2, "d8_flow_accumul": 2, "out_typ": 2, "pntr": 2, "befor": 2, "calcul": [2, 5], "sure": 2, "otherwis": [2, 5], "measur": [2, 3], "express": [2, 5], "just": [2, 3, 5], "over": [2, 5, 6], "2000": [2, 3], "0f": [2, 3, 4, 5], "1e6": [2, 3], "streams_path": 2, "vancouver_island_stream": 2, "extract_stream": 2, "zero_background": 2, "final": [2, 5], "exampl": [2, 3], "deriv": [2, 3, 4, 5], "mostli": 2, "thei": [2, 3, 5, 6], "far": [2, 3], "perfect": 2, "figur": 2, "diverg": 2, "watercours": [2, 3], "defin": 2, "nation": [2, 3], "hydrograph": [2, 3], "nhn": [2, 3], "filter": 2, "squar": 2, "chapter": 2, "light": [2, 3], "blue": [2, 3], "layer": [2, 3, 4, 5, 6], "previou": [3, 5, 6], "input": [3, 4], "waterbodi": 3, "remov": [3, 4, 5], "within": [3, 5], "remain": 3, "serv": 3, "were": 3, "pre": 3, "origin": [3, 5], "cover": 3, "canada": 3, "download": 3, "select": 3, "vancouver_island_lak": 3, "produc": [3, 5], "shown": 3, "yellow": 3, "illustr": 3, "discontinu": 3, "due": 3, "screen": 3, "dem_fold": 3, "dem": [3, 4, 5, 6], "clariti": 3, "releg": 3, "separ": 3, "To": 3, "py": [3, 5], "pour_pt_path": 3, "pour_point": [3, 4], "d8_path": 3, "_d8_pointer": [3, 4], "_acc": 3, "stream_path": 3, "_stream": 3, "stream_rast": 3, "stream_cr": 3, "22x22m": 3, "minimum": [3, 4], "5": [3, 4], "limit": [3, 4, 6], "sake": 3, "min_basin_area": [3, 4], "min": [3, 4], "compris": 3, "basin_threshold": 3, "10103": 3, "numpi": [3, 4, 5], "np": [3, 4, 5], "rt0": 3, "fdir": 3, "acc": 3, "matrix": 3, "rt1": 3, "1f": [3, 4, 5], "8": [3, 5], "5s": 3, "list": [3, 4, 5], "indic": [3, 4], "stream_px": 3, "argwher": [3, 5], "dictionari": [3, 5], "potenti": 3, "iter": [3, 5], "through": [3, 5], "3x3": 3, "than": 3, "toward": 3, "ppt": [3, 4], "nn": 3, "j": [3, 5], "c_idx": 3, "outlet": 3, "definit": 3, "especi": 3, "preval": 3, "coastal": 3, "focus_cell_acc": 3, "focus_cell_dir": 3, "focu": 3, "nan": [3, 5], "also": 3, "conf": 3, "boolean": [3, 5], "centr": 3, "s_w": 3, "max": [3, 4], "copi": [3, 5], "f_w": 3, "focal": 3, "f_m": 3, "mask_flow_direct": 3, "target": 3, "check_for_conflu": 3, "convert": [3, 4], "geodatafram": [3, 4], "same": [3, 5], "output_ppt_path": 3, "_ppt": 3, "t0": [3, 4, 5], "ppt_df": 3, "datafram": [3, 4, 5], "from_dict": 3, "orient": 3, "cell_idx": [3, 4], "reset_index": [3, 4], "inplac": [3, 4], "str": 3, "ix": 3, "jx": 3, "len": [3, 4, 5], "22560": 3, "n_pts_tot": 3, "n_pts_conf": 3, "n_pts_outlet": 3, "Of": 3, "100": 3, "934567": 3, "21044": 3, "1516": 3, "thu": 3, "onli": 3, "still": 3, "appli": 3, "transform": [3, 5], "ppt_gdf": [3, 4], "create_pour_point_gdf": 3, "chunk": 3, "0s": 3, "20": [3, 4], "1s": [3, 5], "30": [3, 5], "40": 3, "2s": 3, "50": 3, "70": 3, "3s": [3, 5], "80": 3, "90": [3, 5], "4s": 3, "in0": 3, "ta": 3, "match": 3, "to_cr": [3, 5], "4617": 3, "tb": 3, "2f": [3, 4, 5], "One": 3, "vestig": 3, "lot": 3, "fall": 3, "netowork": 3, "perman": 3, "unknown": 3, "avail": 3, "intermitt": 3, "water_definit": 3, "label": [3, 4, 5], "No": 3, "type": 3, "canal": 3, "artifici": 3, "navig": 3, "waterwai": 3, "channel": 3, "conduit": 3, "aqueduct": 3, "penstock": 3, "flume": 3, "sluic": 3, "design": 3, "drainag": [3, 5], "ditch": 3, "small": 3, "manmad": 3, "construct": 3, "earth": 3, "rock": 3, "convei": 3, "inland": 3, "consider": 3, "reservoir": 3, "wholli": 3, "partial": [3, 4], "featur": [3, 5], "store": 3, "regul": 3, "control": 3, "6": [3, 4, 5], "tidal": 3, "7": [3, 5], "affect": 3, "tide": 3, "liquid": 3, "wast": 3, "industri": 3, "complex": 3, "though": 3, "season": 3, "darker": 3, "grei": 3, "read": 3, "region_lakes_path": 3, "_lake": 3, "lakes_df": 3, "c": [3, 5], "index_right": 3, "index_left": 3, "assert": [3, 4, 5], "subject": 3, "criteria": 3, "improv": [3, 5], "perform": [3, 4, 5, 6], "discoveri": 3, "01": [3, 5], "speed": 3, "order": [3, 4], "reloc": 3, "mouth": 3, "buffer": [3, 4, 5], "simplifi": [3, 4], "smooth": 3, "reduc": 3, "heavili": 3, "braid": 3, "headwat": 3, "aren": 3, "too": [3, 4], "close": 3, "proxim": 3, "nearest": 3, "cross": 3, "line": [3, 5], "interpol": 3, "few": [3, 5], "miss": [3, 5], "when": [3, 5, 6], "paramet": [3, 5], "consid": 3, "simplif": 3, "elimin": 3, "vertic": 3, "must": 3, "enough": [3, 6], "10000": 3, "lakes_filt": 3, "lake_ppt": 3, "sjoin": 3, "predic": 3, "filtered_ppt": 3, "isna": 3, "19184": 3, "3376": 3, "least": 3, "lakes_with_pt": 3, "filtered_lak": 3, "uniqu": [3, 4, 5], "id": [3, 5], "lake_id": 3, "isin": 3, "contigu": 3, "adjac": 3, "unary_union": 3, "explod": 3, "index_part": 3, "drop": [3, 4, 5], "311": 3, "shift": 3, "method": [3, 5], "best": 3, "align": 3, "well": [3, 4, 6], "unnecessari": 3, "good": 3, "bad": 3, "behaviour": 3, "ad": 3, "being": 3, "unintent": 3, "geometr": [3, 5], "manipul": 3, "modifi": 3, "address": 3, "case": 3, "op": 3, "n": [3, 5], "tot_pt": 3, "min_acc_cel": 3, "points_to_check": 3, "group": [3, 5], "give": 3, "slight": 3, "lake_geom": 3, "multipolygon": [3, 5], "keep": [3, 4, 5], "main": 3, "remaind": 3, "area_1": 3, "loc": [3, 5], "idxmax": 3, "continu": [3, 5], "resampl": 3, "vector": [3, 4, 5], "resampled_shorelin": 3, "redistribute_vertic": [3, 4], "exterior": 3, "coord": 3, "xy": 3, "xs": 3, "tolist": 3, "ys": 3, "closest": 3, "edg": 3, "px_pt": 3, "sel": 3, "toler": 3, "latlon": 3, "zip": [3, 4, 5], "acc_val": 3, "squeez": 3, "item": 3, "298": 3, "582": 3, "150": 3, "942": 3, "1285": 3, "250": 3, "1741": 3, "300": 3, "2011": 3, "2078": 3, "all_pt": 3, "inp": 3, "pt": [3, 4], "both": [3, 5], "avoid": [3, 4, 5], "pt_dist": 3, "width": 3, "min_spac": 3, "dist_check": 3, "accum_check": 3, "accum": 3, "95": 3, "max_acc": 3, "ani": [3, 5], "not_in_any_lak": 3, "sum": [3, 5], "lg": 3, "append": [3, 4, 5], "500": 3, "750": 3, "1000": 3, "1250": 3, "1500": 3, "1750": 3, "rpt": 3, "all_pts_filt": 3, "dist": 3, "ptg": 3, "refer": 3, "updat": [3, 4, 5], "against": 3, "concat": [3, 4], "1668": 3, "400": 3, "600": 3, "800": 3, "1200": 3, "1400": 3, "1600": 3, "output": [3, 4, 5], "new_pt": 3, "axi": [3, 4], "to_fil": [3, 4, 5], "_pour_points3": 3, "12": [4, 5], "computation": 4, "intens": 4, "unnest_basin": 4, "realli": 4, "written": 4, "rust": 4, "mani": 4, "implicitli": 4, "care": 4, "paralel": 4, "effici": 4, "manag": [4, 6], "usag": 4, "stream": 4, "multiprocess": 4, "mp": 4, "make_valid": [4, 5], "raster_to_vector_basins_batch": 4, "send": 4, "huge": 4, "temporari": 4, "could": 4, "easili": 4, "exce": 4, "ssd": 4, "capac": 4, "raster_fnam": 4, "raster_cr": 4, "min_area": 4, "temp_fold": 4, "raster_path": 4, "raster_no": 4, "temp_polygons_": 4, "05": [4, 5], "shp": 4, "non": 4, "overlap": 4, "raster_to_vector_polygon": 4, "gdf": [4, 5], "self": 4, "simplify_dim": 4, "sqrt": [4, 5], "buffer_dim": 4, "filter_and_explode_geom": 4, "return": [4, 5], "fdir_path": 4, "size": [4, 5], "mb": 4, "files": 4, "getsiz": 4, "temp_dir": 4, "temp": 4, "ppt_dir": 4, "basin_output_dir": 4, "d": [4, 5], "match_ppt_to_polygons_by_ord": 4, "check_for_ppt_batch": 4, "create_ppt_file_batch": 4, "clean_up_temp_fil": 4, "region_rast": 4, "region_raster_cr": 4, "raster_resolut": 4, "tupl": 4, "break": 4, "duplic": 4, "ppt_file": 4, "_pour_point": 4, "ppt_cr": 4, "drop_dupl": 4, "subset": [4, 6], "ignore_index": 4, "allow": 4, "track": [4, 5], "output_fpath": 4, "_basin": 4, "batch_dir": 4, "ppt_batch": 4, "batches_exist": 4, "temp_ppt_filepath": 4, "batch_ppt_path": 4, "batch_ppt_fil": 4, "sort": 4, "n_batch_fil": 4, "batch_output_fil": 4, "startswith": 4, "batch_match": 4, "b": [4, 5], "basin_output_fil": 4, "ppt_batch_path": 4, "t_batch_start": 4, "batch_no": 4, "temp_fnam": 4, "temp_rast": 4, "temp_basin_raster_path": 4, "batch_output_fpath": 4, "04d": 4, "default_callback": 4, "batch_rast": 4, "tb1": 4, "sub": 4, "parallel": 4, "crs_arrai": 4, "min_a_arrai": 4, "batch_raster_fpath": 4, "resolution_arrai": 4, "temp_folder_arrai": 4, "path_input": 4, "batch_size_gb": 4, "1e3": 4, "processor": 4, "core": 4, "multipl": 4, "balanc": 4, "estim": 4, "n_proc": 4, "pool": 4, "trv0": 4, "all_polygon": 4, "trv1": 4, "trc0": 4, "batch_polygon": 4, "maintain": 4, "unnest": 4, "sort_valu": 4, "trc1": 4, "delet": 4, "yet": 4, "directli": [4, 6], "concatenat": 4, "basin_fil": 4, "batch_df": 4, "fpath": [4, 5], "all_data": 4, "gpkg": [4, 5], "driver": 4, "t_n": 4, "n_processed_basin": 4, "ut": [4, 5], "american": [5, 6], "latifovic2010north": 5, "2010": [5, 6], "2015": [5, 6], "2020": [5, 6], "captur": 5, "geospati": [5, 6], "soil": 5, "terrain": 5, "trim": 5, "isol": 5, "numba": 5, "jit": 5, "scipi": 5, "stat": 5, "mstat": 5, "gmean": 5, "coverag": 5, "crop": 5, "interest": 5, "nalcms_fpath": 5, "nalcms_fil": 5, "nalcms_rast": 5, "nalcms_cr": 5, "nalcms_affin": 5, "48": 5, "syntaxerror": 5, "invalid": 5, "syntax": 5, "year": 5, "_3005": 5, "output_fold": 5, "geospatial_lay": 5, "nalcms_": 5, "global": 5, "hydrogeolog": 5, "sp2_ttjniu_2018": 5, "permeabl": 5, "poros": 5, "necessari": 5, "glhymps_path": 5, "gldf": 5, "glhymps_output_path": 5, "glhymps_": 5, "batch": [5, 6], "last": 5, "most": 5, "hopefulli": 5, "clear": 5, "basins_fold": 5, "basins_df": 5, "metadata": 5, "catchment": 5, "descript": 5, "aggreg": 5, "unit": 5, "geom": 5, "deg": 5, "centroid": 5, "flag": 5, "lulc_check": 5, "binari": 5, "drainage_area_km2": 5, "elevation_m": 5, "sea": 5, "slope": 5, "slope_deg": 5, "circ": 5, "aspect": 5, "aspect_deg": 5, "circular": 5, "cropland": 5, "land_use_crops_frac_": 5, "forest": 5, "land_use_forest_frac_": 5, "grassland": 5, "land_grass_forest_frac_": 5, "shrub": 5, "land_use_shrubs_frac_": 5, "snow": 5, "ic": 5, "land_use_snow_ice_frac_": 5, "urban": 5, "land_use_urban_frac_": 5, "land_use_water_frac_": 5, "wetland": 5, "land_use_wetland_frac_": 5, "permeability_logk_m2": 5, "porosity_frac": 5, "wsg84": 5, "counter": 5, "clockwis": 5, "east": 5, "suffix": 5, "lulc": 5, "attribute_funct": 5, "clip_raster_to_basin": 5, "check_lulc_sum": 5, "pct": 5, "intuit": 5, "qualiti": 5, "rais": 5, "checksum": 5, "3f": 5, "recategorize_lulc": 5, "land_use_forest_frac": 5, "land_use_shrubs_frac": 5, "11": 5, "grass": 5, "land_use_grass_frac": 5, "9": 5, "13": 5, "land_use_wetland_frac": 5, "land_use_crops_frac": 5, "15": 5, "land_use_urban_frac": 5, "17": 5, "land_use_water_frac": 5, "18": 5, "snow_ic": 5, "land_use_snow_ice_frac": 5, "19": 5, "lulc_dict": 5, "prop_val": 5, "round": 5, "kei": 5, "get_value_proport": 5, "proport": 5, "count": 5, "ratio": 5, "all_val": 5, "flatten": 5, "val": 5, "isnan": 5, "n_pt": 5, "return_count": 5, "prop_dict": 5, "k": 5, "v": 5, "process_lulc": 5, "basin_geom": 5, "basin_polygon": 5, "basin_id": 5, "raster_load": 5, "lu_raster_clip": 5, "verifi": 5, "check_and_repair_geometri": 5, "in_featur": 5, "geodf": 5, "deep": 5, "is_empti": 5, "repair": 5, "broken": 5, "loop": 5, "poli": 5, "except": 5, "valueerror": 5, "process_basin_elev": 5, "clipped_rast": 5, "evalu": 5, "mean_val": 5, "nanmean": 5, "median_v": 5, "nanmedian": 5, "min_val": 5, "nanmin": 5, "max_val": 5, "nanmax": 5, "get_soil_properti": 5, "col": 5, "dissolv": 5, "aggfunc": 5, "shape_area": 5, "calculu": 5, "fraction": 5, "area_frac": 5, "sum_check": 5, "elif": 5, "area_weighted_v": 5, "sign": 5, "neg": 5, "add": 5, "back": 5, "multipli": 5, "tri": 5, "weight": 5, "arithmet": 5, "process_glhymp": 5, "precis": 5, "bound": 5, "distort": 5, "nopython": 5, "process_slope_and_aspect": 5, "el_px": 5, "meaning": 5, "either": 5, "costli": 5, "90x90m": 5, "asdfd": 5, "empty_lik": 5, "tot_p": 5, "tot_q": 5, "e_w": 5, "g": 5, "h": 5, "becaus": 5, "arrai": 5, "val_check": 5, "isfinit": 5, "q": 5, "cell_slop": 5, "180": 5, "pi": 5, "arctan": 5, "arctan2": 5, "calculate_circular_mean_aspect": 5, "ravenpi": 5, "http": 5, "com": 5, "csh": 5, "cwra": 5, "blob": 5, "1b167749cdf5984545f8f79ef7d31246418a3b54": 5, "analysi": 5, "l118": 5, "angl": 5, "sine_mean": 5, "divid": 5, "sin": 5, "radian": 5, "cosine_mean": 5, "co": 5, "vector_mean": 5, "360": 5, "calculate_slope_and_aspect": 5, "accord": 5, "hill": 5, "1981": 5, "arg": 5, "scalar": 5, "asfd": 5, "wkt": 5, "to_wkt": 5, "raster_shap": 5, "rdem_clip": 5, "rd": 5, "rdarrai": 5, "no_data": 5, "nodata": 5, "geotransform": 5, "to_gdal": 5, "ts0": 5, "terrainattribut": 5, "attrib": 5, "slope_degre": 5, "ts2": 5, "asdfsd": 5, "mean_slope_deg": 5, "hundredth": 5, "my": 5, "4f": 5, "rdem": 5, "mean_aspect_deg": 5, "region_dem": 5, "dem_cr": 5, "dem_affin": 5, "test": 5, "slice": 5, "statement": 5, "whole": 5, "n_sampl": 5, "all_basin_data": 5, "basin_data": 5, "iloc": 5, "clip_ok": 5, "clipped_dem": 5, "land_cov": 5, "to_dict": 5, "record": 5, "complat": 5, "permeability_no_permafrost": 5, "mean_el": 5, "median_el": 5, "min_el": 5, "max_el": 5, "t1": 5, "output_fnam": 5, "output_fil": 5, "to_csv": 5, "median": 5, "2018": 5, "intel": 5, "i7": 5, "8850h": 5, "cpu": 5, "60ghz": 5, "proporti": 5, "onal": 5, "3x": 5, "land": 6, "availabel": 6, "better": 6, "high": 6, "volum": 6, "larger": 6, "popul": 6, "geopackag": 6, "memori": 6, "major": 6, "seem": 6, "environ": 6}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"bc": 0, "ungaug": 0, "basin": [0, 4, 5, 6], "databas": 0, "bcub": 0, "introduct": 0, "note": 0, "frequent": 0, "ask": 0, "question": 0, "faq": 0, "licens": 0, "refer": [0, 5], "1": 1, "get": [1, 3], "start": 1, "depend": 1, "environ": 1, "dem": [1, 2], "acquisit": 1, "url": 1, "list": 1, "download": [1, 5], "cover": [1, 5], "tile": 1, "set": 1, "view": 1, "file": [1, 4], "gdal": 1, "2": 2, "digit": 2, "elev": 2, "preprocess": 2, "clip": [2, 5], "hydraul": 2, "condit": 2, "flow": [2, 3], "direct": [2, 3, 5], "accumul": [2, 3], "stream": [2, 3], "network": [2, 3], "3": 3, "pour": 3, "point": 3, "extract": [3, 5, 6], "import": 3, "raster": 3, "defin": 3, "confluenc": 3, "filter": 3, "spuriou": 3, "water": 3, "bodi": 3, "geometri": 3, "contain": 3, "find": 3, "add": 3, "lake": 3, "inflow": 3, "4": 4, "delin": 4, "concaten": 4, "batch": 4, "polygon": 4, "singl": 4, "geopackag": 4, "option": 4, "5": 5, "attribut": [5, 6], "nalcm": 5, "data": 5, "glhymp": 5, "retriev": 5, "land": 5, "6": 6, "postgr": 6, "postgi": 6, "method": 6, "instal": 6}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 56}})
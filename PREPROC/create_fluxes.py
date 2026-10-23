import argparse
import numpy as np 
import xarray as xr
import yaml
import matplotlib.pyplot as plt
import define_isobath, plot_control_surfaces, define_arbitrary_transect

'''
adapted from ogstm/testcase/create_fluxes.py
generates diverse *.nc files that define control surfaces to save fluxes at
then saves a single Fluxes.nc with all the control surfaces.
Fluxes.nc is used by OGSTM to save fluxes.
The control surface files can be used
to retrieve specific fluxes from model output.

TODO:
  - code plot zonal, meridional, arbitrary transects

Usage:
source /g100_work/OGS23_PRACE_IT/COPERNICUS/sequence.sh
python create_fluxes.py -y create_fluxes.yaml
'''

def load_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yamlfile", "-y",
        required=True,
        help="path to file with all parameters"
    )
    args = parser.parse_args()
    yamlfile = args.yamlfile
    # yamlfile = create_fluxes.yaml
    with open(yamlfile) as f:
        return yaml.safe_load(f)


def read_masks(Params):
    DM = {}
    # bounmask contains sequential index of all water points
    B = xr.open_dataset(Params["bounmask"])
    idxmap = B["index"].values[0]
    idxmap = np.nan_to_num(idxmap, nan=0).astype(int)
    B.close()
    # meshmask is self explaining
    M = xr.open_dataset(Params["meshmask"])
    tmask = M["tmask"].values[0]
    jpk, jpj, jpi = tmask.shape
    mbathy = tmask.sum(axis=0).astype(int)
    z = M["nav_lev"].values[:]
    glamt = M["glamt"].values[(0, 0)]
    gphit = M["gphit"].values[(0, 0)]
    glamf = M["glamf"].values[(0, 0)]
    gphif = M["gphif"].values[(0, 0)]
    M.close()
    return (idxmap, tmask, jpk, jpj, jpi, mbathy, z, glamt, gphit, glamf, gphif)


def dump_fluxes_ogstm(vals, outdir, outfile, description):
    # saves the Fluxes.nc file that goes in the rundir
    outfile = f"{outdir}/{outfile}"
    O = xr.Dataset(data_vars={"index": ("n", vals)})
    O["index"].attrs["description"] = description
    print(f"saving: {outfile}")
    O.to_netcdf(outfile)
    return


def is_kept_by_lon_cut(lon, Ctx):
    # True where the T-point longitude is east of lon_cut (always True if there is no cut)
    lon = np.asarray(lon)
    if Ctx["lon_cut"] is None:
        return np.ones(lon.shape, dtype=bool)
    return lon >= Ctx["lon_cut"]


def apply_lon_cut(vals, p_U, p_V, p_W, Ctx):
    # removes the grid points west of lon_cut (T-point longitude of the cell)
    # the cut is applied to the control surfaces just before they are saved, 
    # so it applies to all of them and to Fluxes.nc
    vals = np.asarray(vals, dtype=np.int64)
    keep = is_kept_by_lon_cut(Ctx["lon_of_index"][vals], Ctx)
    if Ctx["lon_cut"] is not None:
        print(f"lon_cut = {Ctx['lon_cut']}: removed {int((~keep).sum())} of {len(vals)} points")
    return vals[keep], np.asarray(p_U)[keep], np.asarray(p_V)[keep], np.asarray(p_W)[keep]


def apply_lon_cut_contours(contours, Ctx):
    # same cut as apply_lon_cut, for paths of grid faces (isobath, arbitrary transect): 
    # an edge (U|V, j, i[, sign]) belongs to the T cell (j, i). 
    # Applied before saving, plotting and extracting the points of the isobath, 
    # which then all use the same edges. 
    # Loops are split where edges are removed, so that every contour stays a connected sequence.
    if Ctx["lon_cut"] is None:
        return contours
    cut = []
    n_before = n_after = 0
    for contour in contours:
        segment = []
        for edge in contour:
            n_before += 1
            if is_kept_by_lon_cut(Ctx["glamt"][edge[1], edge[2]], Ctx):
                segment.append(edge)
                n_after += 1
            elif segment:
                cut.append(segment)
                segment = []
        if segment:
            cut.append(segment)
    print(f"lon_cut = {Ctx['lon_cut']}: removed {n_before - n_after} of {n_before} edges")
    return cut


def surface_filename(Surf_Params, ext=".nc", contour_file=False):
    # output file name of a control surface or contours if contour_file=True
    # plus the optional user defined "name" to tell apart surfaces of the same type
    surf_type = Surf_Params.get("type")
    name = Surf_Params.get("name")
    fname = f"CtrlSurf_{surf_type}_{name}{ext}" if name else f"CtrlSurf_{surf_type}{ext}"
    fname = fname.replace('CtrlSurf', 'Contours') if contour_file else fname
    return fname


def dump_fluxes_surf(vals, p_U, p_V, p_W, Ctx, outfile, description, uv_description=None):
    # saves single control surfaces .nc files
    # returns the indexes of the points that are kept after the lon_cut
    if outfile in Ctx["outfiles"]:
        raise ValueError(f"{outfile} would be overwritten by another control surface: "
                         "use the 'name' argument to give them different names")
    Ctx["outfiles"].add(outfile)
    outfile = f"{Ctx['outdir']}/{outfile}"
    vals, p_U, p_V, p_W = apply_lon_cut(vals, p_U, p_V, p_W, Ctx)
    O = xr.Dataset(data_vars={'index': ("n", vals),  
                              'U': ("n", p_U), 
                              'V': ("n", p_V), 
                              'W': ("n", p_W),
                              'lon': ("n", Ctx["lon_of_index"][vals]),
                              'lat': ("n", Ctx["lat_of_index"][vals])})
    O["index"].attrs["description"] = description
    O["U"].attrs["description"] = uv_description or "1 if flux on grid U, else 0"
    O["V"].attrs["description"] = uv_description or "1 if flux on grid V, else 0"
    O["W"].attrs["description"] = "1 if flux on grid W, else 0"
    O["lon"].attrs["description"] = "longitude of the T point of the cell"
    O["lat"].attrs["description"] = "latitude of the T point of the cell"
    print(f"saving: {outfile}")
    O.to_netcdf(outfile)
    return vals.tolist()


def append_indices(idxlist_0, idxlist_1):
    idxlist = list(set(idxlist_0) | set(idxlist_1))
    return idxlist


def compute_depth_level(idxlist, Ctx, Surf_Params):
    # a horizontal control surface at a fixed depth / z level
    # does not account for time-varying z
    idxmap, tmask, jpj, jpi, z = (Ctx["idxmap"], Ctx["tmask"], Ctx["jpj"], Ctx["jpi"], Ctx["z"])
    outdir = Ctx["outdir"]
    depth = Surf_Params["depth"]
    iz = np.abs(z - depth).argmin()
    idxcurr = []
    for j in range(1, jpj - 1):
        for i in range(1, jpi - 1):
            if tmask[(iz, j, i)] == 1:
                idxcurr.append(int(idxmap[(iz, j, i)]))
    #
    p_U = [0] * len(idxcurr)
    p_V = [0] * len(idxcurr)
    p_W = [1] * len(idxcurr)
    depthstr = str(int(depth))
    description = f"indexes of an horizontal surface at {depthstr}m depth"
    outfile = surface_filename(Surf_Params)
    idxcurr = dump_fluxes_surf(idxcurr, p_U, p_V, p_W, Ctx, outfile, description)
    idxlist = append_indices(idxlist, idxcurr)
    return idxlist


def compute_sea_bed(idxlist, Ctx, Surf_Params):
    # the seabed / ocean bottom / ocean floor, 
    # for gravitational settling fluxes
    idxmap, jpj, jpi, mbathy = (Ctx["idxmap"], Ctx["jpj"], Ctx["jpi"], Ctx["mbathy"])
    outdir = Ctx["outdir"]
    idxcurr = []
    for j in range(1, jpj - 1):
        for i in range(1, jpi - 1):
            nlev = mbathy[(j, i)]
            if nlev == 0:
                pass
            else:
                bottom_k = nlev - 1
                idxcurr.append(int(idxmap[(bottom_k, j, i)]))
    #
    p_U = [0] * len(idxcurr)
    p_V = [0] * len(idxcurr)
    p_W = [1] * len(idxcurr)
    outfile = surface_filename(Surf_Params)
    description = "indexes sea bed level water points (the ocean floor)"
    idxcurr = dump_fluxes_surf(idxcurr, p_U, p_V, p_W, Ctx, outfile, description)
    idxlist = append_indices(idxlist, idxcurr)
    return idxlist


def compute_isobath(idxlist, Ctx, Surf_Params):
    # a vertical control surface (surface to seabed) define along an isobath
    # meant to evaluate coastal - open ocean exchange
    # does not account for time-varying z levels.
    idxmap = Ctx["idxmap"]
    outdir = Ctx["outdir"]
    meshmask_file = Ctx["meshmask_file"]
    isobath_depth = Surf_Params["depth"]
    isobath = define_isobath.compute_isobath(meshmask_file, isobath_depth, verbose=True)
    contours = apply_lon_cut_contours(isobath["contours"], Ctx)
    p_i, p_j, grid_type = define_isobath.contours_to_points(contours)
    depthstr = str(int(isobath_depth))
    if Surf_Params.get("save_isobath", False):
        # this output is not used currently, 
        # but keeping it as it may turn out to be useful for some plots and/or analysis later on
        isobath_outfile = f"{outdir}/" + surface_filename(Surf_Params, contour_file=True)
        define_isobath.write_output(isobath_outfile, contours, isobath["glamu"], isobath["gphiu"], isobath["glamv"], isobath["gphiv"], isobath_depth)
    if Surf_Params.get("plot_isobath", False):
        plot_outfile = f"{outdir}/" + surface_filename(Surf_Params, ".png")
        plot_control_surfaces.plot_control_surface_map(contours, (isobath["bathy"]), (isobath["glamt"]), (isobath["gphit"]), (isobath["glamf"]),
          (isobath["gphif"]), title=f"{isobath_depth:.0f} m isobath",
          output=plot_outfile)
    bathy = isobath["bathy"]
    deep = np.isfinite(bathy) & (bathy >= isobath_depth)
    idxcurr = []
    p_V = []
    p_U = []
    for i, j, gt in zip(p_i, p_j, grid_type):
        # Face is shared with T(j,i+1) for U edges, T(j+1,i) for V edges.
        if gt == 0:
            sign = 1 if (not deep[j, i] and deep[j, i + 1]) else -1
        else:
            sign = 1 if (not deep[j, i] and deep[j + 1, i]) else -1
        for x in idxmap[:, j, i]:
            if x == 0:
                continue
            if x not in idxcurr:
                idxcurr.append(x)
                p_U.append(0)
                p_V.append(0)
            k = idxcurr.index(x)
            if gt == 0:
                p_U[k] = sign
            else:
                p_V[k] = sign

    p_W = [0] * len(idxcurr)
    outfile = surface_filename(Surf_Params)
    description = f"indexes of coast - open ocean interface, i.e. along the {depthstr}m isobath"
    uv_description = ("flux direction across the face: 1 if coast to open ocean, "
                      "-1 if open ocean to coast, 0 if the face is not on this grid")
    idxcurr = dump_fluxes_surf(idxcurr, p_U, p_V, p_W, Ctx, outfile, description, uv_description)
    idxlist = append_indices(idxlist, idxcurr)
    return idxlist


def compute_not_implemented(idxlist, Ctx, Surf_Params):
    raise NotImplementedError(f"""control surface type '{Surf_Params["type"]}' is not implemented yet""")


def nearest_face_index(coord_t_line, target):
    k0 = int(np.abs(coord_t_line - target).argmin())
    face = k0 if coord_t_line[k0] <= target else k0 - 1
    return max(0, min(face, len(coord_t_line) - 2))


def compute_zonal_transect(idxlist, Ctx, Surf_Params):
    # a zonal transect at a fixed latitude
    # assumes rectilinear regular grid
    idxmap = Ctx["idxmap"]
    glamt, gphit = Ctx["glamt"], Ctx["gphit"]
    jpi = Ctx["jpi"]
    outdir = Ctx["outdir"]
    lat = Surf_Params["lat"]
    lon_min, lon_max = sorted((Surf_Params["lon_start"], Surf_Params["lon_end"]))
    j = nearest_face_index(gphit[:, 0], lat)
    i = np.arange(1, jpi - 1)
    i = i[(glamt[j, 1:jpi - 1] >= lon_min) & (glamt[j, 1:jpi - 1] <= lon_max)]
    idxcurr = idxmap[:, j, i].T.ravel()
    idxcurr = idxcurr[idxcurr != 0].tolist()
    p_U = [0] * len(idxcurr)
    p_V = [1] * len(idxcurr)
    p_W = [0] * len(idxcurr)
    outfile = surface_filename(Surf_Params)
    description = "indexes of a zonal transect at lat {:.2f}, from surface to bottom".format(lat)
    idxcurr = dump_fluxes_surf(idxcurr, p_U, p_V, p_W, Ctx, outfile, description)
    idxlist = append_indices(idxlist, idxcurr)
    return idxlist


def compute_meridional_transect(idxlist, Ctx, Surf_Params):
    # a meridional transect at a fixed longitude
    # assumes rectilinear regular grid
    idxmap = Ctx["idxmap"]
    glamt, gphit = Ctx["glamt"], Ctx["gphit"]
    jpj = Ctx["jpj"]
    outdir = Ctx["outdir"]
    lon = Surf_Params["lon"]
    lat_min, lat_max = sorted((Surf_Params["lat_start"], Surf_Params["lat_end"]))
    i = nearest_face_index(glamt[0, :], lon)
    j = np.arange(1, jpj - 1)
    j = j[(gphit[1:jpj - 1, i] >= lat_min) & (gphit[1:jpj - 1, i] <= lat_max)]
    idxcurr = idxmap[:, j, i].T.ravel()
    idxcurr = idxcurr[idxcurr != 0].tolist()
    p_U = [1] * len(idxcurr)
    p_V = [0] * len(idxcurr)
    p_W = [0] * len(idxcurr)
    outfile = surface_filename(Surf_Params)
    description = "indexes of a meridional transect at lon {:.2f}, from surface to bottom".format(lon)
    idxcurr = dump_fluxes_surf(idxcurr, p_U, p_V, p_W, Ctx, outfile, description)
    idxlist = append_indices(idxlist, idxcurr)
    return idxlist


def compute_arbitrary_transect(idxlist, Ctx, Surf_Params):
    # a transect from the sea surface to the seabed along a polyline through arbitrary (lon, lat) points,
    # see define_arbitrary_transect.py. Fluxes are positive from the left to the right of the transect,
    # looking from the first to the last point.
    points = Surf_Params["points"]
    T = define_arbitrary_transect.compute_arbitrary_transect(
        points, Ctx["glamf"], Ctx["gphif"], Ctx["idxmap"], verbose=True)
    if Surf_Params.get("plot_transect", True):
        # the plot shows the path that is kept after the lon_cut, as for the isobath
        contours = apply_lon_cut_contours([T["edges"]], Ctx)
        (bathy, tmask, glamt, gphit, glamu, gphiu, glamv, gphiv, glamf, gphif) = define_isobath.read_meshmask(Ctx["meshmask_file"])
        plot_outfile = f"{Ctx['outdir']}/" + surface_filename(Surf_Params, ".png")
        plot_control_surfaces.plot_control_surface_map(
            contours, bathy, glamt, gphit, glamf, gphif,
            title=f"arbitrary transect {Surf_Params.get('name', '')}".strip(),
            output=plot_outfile)
        plt.close("all")
    outfile = surface_filename(Surf_Params)
    description = "indexes of an arbitrary transect through {}, from surface to bottom".format(
        " - ".join("({}, {})".format(lon, lat) for lon, lat in points))
    uv_description = ("flux direction across the face: 1 if the positive flux is from the left to the right "
                      "of the transect (looking from the first to the last point), -1 if from the right to the left, "
                      "0 if the face is not on this grid")
    idxcurr = dump_fluxes_surf(T["index"], T["U"], T["V"], T["W"], Ctx, outfile, description, uv_description)
    idxlist = append_indices(idxlist, idxcurr)
    return idxlist


if __name__ == "__main__":
    Params = load_parameters()
    DCS = Params["Control_Surfaces"]
    outdir = Params["outdir"]
    idxmap, tmask, jpk, jpj, jpi, mbathy, z, glamt, gphit, glamf, gphif = read_masks(Params)
    # T-point lon, lat of each bounmask index (to apply lon_cut and to save in the output files)
    kk, jj, ii = np.nonzero(idxmap)
    lon_of_index = np.full(idxmap.max() + 1, np.nan)
    lat_of_index = np.full(idxmap.max() + 1, np.nan)
    lon_of_index[idxmap[kk, jj, ii]] = glamt[jj, ii]
    lat_of_index[idxmap[kk, jj, ii]] = gphit[jj, ii]
    Ctx = {'idxmap':idxmap, 
           'tmask':tmask, 
           'jpk':jpk, 
           'jpj':jpj,  'jpi':jpi,  'mbathy':mbathy, 
           'z':z, 
           'glamt':glamt, 
           'gphit':gphit, 
           'glamf':glamf, 
           'gphif':gphif, 
           'lon_of_index':lon_of_index, 
           'lat_of_index':lat_of_index, 
           'lon_cut':Params.get("lon_cut"), 
           'outfiles':set(), 
           'meshmask_file':Params["meshmask"], 
           'outdir':outdir}
    function_handlers = {'depth_level': compute_depth_level, 
                         'sea_bed': compute_sea_bed, 
                         'zonal_transect': compute_zonal_transect, 
                         'meridional_transect': compute_meridional_transect, 
                         'arbitrary_transect': compute_arbitrary_transect, 
                         'isobath': compute_isobath}
    # Compute control surfaces
    idxlist = []
    for key, Surf_Params in DCS.items():
        ftype = Surf_Params["type"]
        curr_f = function_handlers.get(ftype)
        if curr_f is None:
            raise ValueError(f"{key}: unrecognised control surface type '{ftype}'")
        print(f"--- {key}: {ftype} ---")
        idxlist = curr_f(idxlist, Ctx, Surf_Params)
    # Save combined output
    dump_fluxes_ogstm(idxlist, outdir, "Fluxes.nc", "indexes of all the water points where to save fluxes at")

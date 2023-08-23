# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 07:41:32 2019

Deterministic nowcast with pySTEPS, with extraction of results per catchment. 
Based on the input data for the Ensemble nowcast, but without any ensembles. 

Make sure to change the initial part to your case.

Note that this script assumes that the catchments are already reprojected.

TO DO - add _reprojected to input and change this later on in the script.

@author: imhof_rn
"""

from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr, osr

import os
#os.environ['PROJ_LIB'] = r'/u/imhof_rn/anaconda3/pkgs/proj4-5.2.0-h470a237_1/share/proj'
os.chdir('/users/junzheyin/Large_Sample_Nowcasting_Evaluation/pysteps')
import mkl
mkl.set_num_threads(1)

import datetime
import netCDF4
import numpy as np
import pprint
import sys
import time

import pysteps as stp
import config as cfg

import logging
import itertools

logging.basicConfig(level=logging.INFO)

# import message passing interface for python
from mpi4py import MPI

# import for memory use
#from pympler import tracker
#tr = tracker.SummaryTracker()
#tr.print_diff() 

###############################################################################
#################
# Initial part, only change this
# NOTE: This script only works when the catchment shapefiles are already reprojected
# to the KNMI radar dataset.
#################

#os.chdir('/u/imhof_rn/pysteps-0.2')

# Catchment filenames and directories
catchments = False # Put on false when you don't want any slicing for catchments (i.e. you will use the full output)
# If catchments = 'False', uncomment the next two lines.
catchment_filenames = ["/bulk/junzheyin/catchmentss/Hupsel.shp", "/bulk/junzheyin/catchments/stroomgebied_Regge.shp", "/bulk/junzheyin/catchments/GroteWaterleiding.shp", "/bulk/junzheyin/catchments/Aa.shp", "/bulk/junzheyin/catchments/Reusel.shp", "/bulk/junzheyin/catchments/het_molentje.shp", "/bulk/junzheyin/catchments/Luntersebeek.shp", "/bulk/junzheyin/catchments/Dwarsdiep.shp", "/bulk/junzheyin/catchments/AfwaterendgebiedBoezemsysteem.shp", "/bulk/junzheyin/catchments/HHRijnland.shp", "/bulk/junzheyin/catchments/Beemster.shp", "/bulk/junzheyin/catchments/DeLinde.shp"] # Put here the locations of the shapefiles
catchment_names = ['Hupsel', 'Regge', 'GroteWaterleiding', 'Aa', 'Reusel', 'Molentje', 'Luntersebeek', 'Dwarsdiep', 'Delfland', 'Rijnland', 'Beemster', 'Linde'] # A list of catchment names.
out_dir = "/users/junzheyin/Large_Sample_Nowcasting_Evaluation/pysteps" # Just used for logging, the actual
# out_dir is set in the pystepsrc-file.

# Verification settings
verification = {
    "experiment_name"   : "pysteps_mpi_24hours_deterministic",
    "overwrite"         : True,            # to recompute nowcasts
    "v_thresholds"      : [0.1, 1.0],       # [mm/h]                 
    "v_leadtimes"       : [10, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],     # [min]
    "v_accu"            : None,             # [min]
    "seed"              : 42,               # for reproducibility
    "doplot"            : True,            # save figures
    "dosaveresults"     : True              # save verification scores to csv
}

# Forecast settings
forecast = {
    "n_lead_times"      : 72,       # timesteps per nowcast
    "r_threshold"       : 0.1,      # rain/no rain threshold [mm/h]
    "unit"              : "mm/h",   # mm/h or dBZ
    "transformation"    : "dB",     # None or dB 
    "adjust_domain"     : None      # None or square
}

# The experiment set-up
## this includes tuneable parameters
experiment = {
    ## the events           event start     event end       update cycle  data source
    "data"              : [('201508131920', '201508132220', 5, 'knmi'), 
      ('201508132000', '201508132300', 5, 'knmi'), 
      ('201508261735', '201508262035', 5, 'knmi'), 
      ('201508261800', '201508262100', 5, 'knmi'), 
      ('201509041755', '201509042055', 5, 'knmi'), 
      ('201509041820', '201509042120', 5, 'knmi'), 
      ('201605301655', '201605301955', 5, 'knmi'), 
      ('201605301745', '201605302045', 5, 'knmi'), 
      ('201605301800', '201605302100', 5, 'knmi'), 
      ('201606201210', '201606201510', 5, 'knmi'), 
      ('201606201300', '201606201600', 5, 'knmi'), 
      ('201606222325', '201606230225', 5, 'knmi'), 
      ('201606230050', '201606230350', 5, 'knmi'), 
      ('201606230100', '201606230400', 5, 'knmi'), 
      ('201606230200', '201606230500', 5, 'knmi'), 
      ('201606230300', '201606230600', 5, 'knmi'), 
      ('201707061955', '201707062255', 5, 'knmi'), 
      ('201707062000', '201707062300', 5, 'knmi'), 
      ('201707120455', '201707120755', 5, 'knmi'), 
      ('201707120520', '201707120820', 5, 'knmi'), 
      ('201707120600', '201707120900', 5, 'knmi'), 
      ('201707120700', '201707121000', 5, 'knmi'), 
      ('201707291755', '201707292055', 5, 'knmi'), 
      ('201707291800', '201707292100', 5, 'knmi'), 
      ('201708300055', '201708300355', 5, 'knmi'), 
      ('201708300155', '201708300455', 5, 'knmi'), 
      ('201708300255', '201708300555', 5, 'knmi'), 
      ('201708300300', '201708300600', 5, 'knmi'), 
      ('201708300420', '201708300720', 5, 'knmi'), 
      ('201708300500', '201708300800', 5, 'knmi'), 
      ('201709081455', '201709081755', 5, 'knmi'), 
      ('201709081545', '201709081845', 5, 'knmi'), 
      ('201709081650', '201709081950', 5, 'knmi'), 
      ('201709081725', '201709082025', 5, 'knmi'), 
      ('201709081800', '201709082100', 5, 'knmi'), 
      ('201709081900', '201709082200', 5, 'knmi'), 
      ('201709141055', '201709141355', 5, 'knmi'), 
      ('201709141155', '201709141455', 5, 'knmi'), 
      ('201709141200', '201709141500', 5, 'knmi'), 
      ('201709141300', '201709141600', 5, 'knmi'), 
      ('201711270745', '201711271045', 5, 'knmi'), 
      ('201804102045', '201804102345', 5, 'knmi'), 
      ('201804102100', '201804110000', 5, 'knmi'), 
      ('201804102200', '201804110100', 5, 'knmi'), 
      ('201804292255', '201804300155', 5, 'knmi'), 
      ('201804292310', '201804300210', 5, 'knmi'), 
      ('201804300000', '201804300300', 5, 'knmi'), 
      ('201805291455', '201805291755', 5, 'knmi'), 
      ('201805291515', '201805291815', 5, 'knmi'), 
      ('201805291600', '201805291900', 5, 'knmi'), 
      ('201808101755', '201808102055', 5, 'knmi'), 
      ('201808101820', '201808102120', 5, 'knmi'), 
      ('201808101930', '201808102230', 5, 'knmi'), 
      ('201808102000', '201808102300', 5, 'knmi'), 
      ('201808242050', '201808242350', 5, 'knmi'), 
      ('201808242100', '201808250000', 5, 'knmi'), 
      ('201808242240', '201808250140', 5, 'knmi'), 
      ('201808242300', '201808250200', 5, 'knmi'), 
      ('201809050545', '201809050845', 5, 'knmi'), 
      ('201809050600', '201809050900', 5, 'knmi'), 
      ('201809050700', '201809051000', 5, 'knmi'), 
      ('201810300150', '201810300450', 5, 'knmi'), 
      ('201810300200', '201810300500', 5, 'knmi'), 
      ('201810300300', '201810300600', 5, 'knmi'), 
      ('201906052100', '201906060000', 5, 'knmi'), 
      ('201906052200', '201906060100', 5, 'knmi'), 
      ('201906052300', '201906060200', 5, 'knmi'), 
      ('201906120755', '201906121055', 5, 'knmi'), 
      ('201906120820', '201906121120', 5, 'knmi'), 
      ('201906120940', '201906121240', 5, 'knmi'), 
      ('201906121000', '201906121300', 5, 'knmi'), 
      ('201906150155', '201906150455', 5, 'knmi'), 
      ('201906150240', '201906150540', 5, 'knmi'), 
      ('201906150300', '201906150600', 5, 'knmi'), 
      ('201906150400', '201906150700', 5, 'knmi'), 
      ('201910061250', '201910061550', 5, 'knmi'), 
      ('201910210430', '201910210730', 5, 'knmi'), 
      ('202002091830', '202002092130', 5, 'knmi'), 
      ('202002091900', '202002092200', 5, 'knmi'), 
      ('202006050545', '202006050845', 5, 'knmi'), 
      ('202006050600', '202006050900', 5, 'knmi'), 
      ('202006122025', '202006122325', 5, 'knmi'), 
      ('202006171755', '202006172055', 5, 'knmi'), 
      ('202006171855', '202006172155', 5, 'knmi'), 
      ('202006171920', '202006172220', 5, 'knmi'), 
      ('202006172000', '202006172300', 5, 'knmi'), 
      ('202007251955', '202007252255', 5, 'knmi'), 
      ('202007252055', '202007252355', 5, 'knmi'), 
      ('202007252100', '202007260000', 5, 'knmi'), 
      ('202008161555', '202008161855', 5, 'knmi'), 
      ('202008161600', '202008161900', 5, 'knmi'), 
      ('202009232020', '202009232320', 5, 'knmi'), 
      ('202009232100', '202009240000', 5, 'knmi')],
     
                                                                                                                                                
    ## the methods
    "oflow_method"      : ["lucaskanade"],      # lucaskanade, darts
    "adv_method"        : ["semilagrangian"],   # semilagrangian, eulerian
    "nwc_method"        : ["steps"],
    "noise_method"      : [None],    # parametric, nonparametric, ssft
    "decomp_method"     : ["fft"],
    
    ## the parameters
    "n_ens_members"     : [1],
    "ar_order"          : [2],
    "n_cascade_levels"  : [8],
    "noise_adjustment"  : [False],
    "conditional"       : [False],
    "precip_mask"       : [True],
    "mask_method"       : ["sprog"],      # obs, incremental, sprog
    "prob_matching"     : ["mean"],
    "num_workers"       : [1],         # Set the number of processors available for parallel computing
    "vel_pert_method"   : [None],       # No velocity pertubation in order to allow for deterministic run following Seed et al. [2003]
}

# End of initial part
###############################################################################

start_time = time.time()

#### HERE ALL AVAILABLE PROCESSES AT START-UP TIME ARE COLLECTED IN comm
#### SEE FOR MORE INFO ON MPI: https://www.cs.earlham.edu/~lemanal/slides/mpi-slides.pdf 
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

logging.info(('I am process rank {}'.format(rank)))

#########################################################
# Open the catchment shapes - They're needed later for the catchment_slice utils
#########################################################
shapes = []

for i in range(0, len(catchment_filenames)):
    shape_filename = catchment_filenames[i]
    
    # set file names in order to obtain the reprojected shapefile, which 
    # was made with the catchment_medata functionality.
    dirname = os.path.dirname(shape_filename)
    basename = os.path.basename(shape_filename)
    basenametxt = os.path.splitext(basename)[0]
    shapes_reprojected = os.path.join(dirname, basenametxt+'_Reprojected.shp')	
    
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shapes.append(driver.Open(shapes_reprojected))

###########
# Set some first functions
###########

## define the callback function to export the nowcast to netcdf
converter   = stp.utils.get_method("mm/h")
def export(X):
    ## convert to mm/h
    X,_ = converter(X, metadata)
    # readjust to initial domain shape
    X,_ = reshaper(X, metadata, inverse=True)
    # Then, slice the array per catchment or not if no catchments are given
    if catchments == True:
        X_catchment = stp.utils.catchment_slice_mpi(X, shapes)
        # Export to netCDF per catchment
        for n in range(0, len(catchment_filenames)):
            key = list(d.keys())[n]
            stp.io.export_forecast_dataset(X_catchment[n], d[key])
    else:
        # else, export full radar nowcast to netcdf
        stp.io.export_forecast_dataset(X, exporter)

# Conditional parameters
## parameters that can be directly related to other parameters
def cond_pars(pars):
    for key in list(pars):
        if key == "oflow_method":
            if pars[key].lower() == "darts":  pars["n_prvs_times"] = 9
            else:                             pars["n_prvs_times"] = 3
        elif key.lower() == "n_cascade_levels":
            if pars[key] == 1 : pars["bandpass_filter"] = "uniform"
            else:               pars["bandpass_filter"] = "gaussian"
        elif key.lower() == "nwc_method":
            if pars[key] == "extrapolation" : pars["n_ens_members"] = 1
    return pars

#########
# Make list of parameters (i.e. the different dates - all other parameters are
# the same for every run) and scatter these over the nodes.
#########
    
# Prepare the list of all parameter sets of the verification
parsets = [[]]
for _, items in experiment.items():
    parsets = [parset+[item] for parset in parsets for item in items]

if rank == 0:
    #### Reorganize work a bit so we can scatter it
    keyfunc = lambda x:x[0] % size
    work = itertools.groupby(sorted(enumerate(parsets), key=keyfunc), keyfunc)
    
    #### Expand the work so we get lists of row, col per node
    workpernode = [[x[1] for x in val] for (key, val) in work]
else:
    workpernode = None

#### NOW DISTRIBUTE THE WORK
workpernode = comm.scatter(workpernode, root=0)

logging.info("Got the following work in process rank {} : {}".format(rank, workpernode))

#### Each node can now do it's own work. The main advantage is that we can do a gather at the end to collect all results.
#### Keep track of all the runs per node in scores
#scores = []

#### before starting any runs, make sure that you know in which folder we run this MPI run routine. 
#### Always return to this folder before the next run
#curdir = os.getcwd()
#os.chdir('/u/imhof_rn/pysteps-master')

###########
# Run the model in parallel
###########

# Now loop all parameter sets
for n, parset in enumerate(workpernode):
#    logging.info("rank %02.f computing scores for parameter set nr %04.f" % (rank, n))
    runId = '%s_%04.f' % (out_dir, n)
    
    # Build parameter set
    
    p = {}
    for m, key in enumerate(experiment.keys()):
        p[key] = parset[m]
    ## apply conditional parameters
    p = cond_pars(p)
    ## include all remaining parameters
    p.update(verification)
    p.update(forecast)
    
#    print("************************")
#    print("* Parameter set %02d/%02d: *" % (n+1, len(parsets)))
#    print("************************")
    
#    pprint.pprint(p)
    
    # If necessary, build path to results
    path_to_experiment = os.path.join(cfg.path_outputs, p["experiment_name"])
    # subdir with event date
    path_to_nwc = os.path.join(path_to_experiment, '-'.join([p["data"][0], p["data"][3]]))
#    for key, item in p.items():
#		# include only variables that change
#        if len(experiment.get(key,[None])) > 1 and key.lower() is not "data":
#            path_to_nwc = os.path.join(path_to_nwc, '-'.join([key, str(item)]))
    try:
        os.makedirs(path_to_nwc)
    except OSError:
        pass
        
    # **************************************************************************
    # NOWCASTING
    # ************************************************************************** 
    
    # Loop forecasts within given event using the prescribed update cycle interval

    ## import data specifications
    ds = cfg.get_specifications(p["data"][3])
    
    if p["v_accu"] is None:
        p["v_accu"] = ds.timestep
    
    # Loop forecasts for given event
    startdate   = datetime.datetime.strptime(p["data"][0], "%Y%m%d%H%M")
    enddate     = datetime.datetime.strptime(p["data"][1], "%Y%m%d%H%M")
    countnwc = 0
    while startdate <= enddate:
            # filename of the nowcast netcdf. Set name either per catchment or as 
            # total nowcast for the entire radar image.
            if catchments == True:
                outfn = []
                for n in range(0, len(catchment_names)):
                    path_to_catchment = os.path.join(path_to_nwc, catchment_names[n])
                    try:
                        os.makedirs(path_to_catchment)
                        Name = os.path.join(path_to_catchment, "%s_nowcast.netcdf" % startdate.strftime("%Y%m%d%H%M"))
                        outfn.append(Name)
                    except OSError:
                        print("Catchment outfile directory does already exist for starttime: %s" % startdate.strftime("%Y%m%d%H%M"))
                        Name = os.path.join(path_to_catchment, "%s_nowcast.netcdf" % startdate.strftime("%Y%m%d%H%M"))
                        outfn.append(Name)
            else:
                outfn = os.path.join(path_to_nwc, "%s_nowcast.netcdf" % startdate.strftime("%Y%m%d%H%M"))
        
            ## check if results already exists
            if catchments == True:
                run_exist = False
                if os.path.isfile(outfn[n]):
                    fid = netCDF4.Dataset(outfn[n], 'r')
                    if fid.dimensions["time"].size == p["n_lead_times"]:
                        run_exist = True
                        if p["overwrite"]:
                            os.remove(outfn[n])
                            run_exist = False    
                    else:
                        os.remove(outfn[n])
            else:
                run_exist = False
                if os.path.isfile(outfn):
                    fid = netCDF4.Dataset(outfn, 'r')
                    if fid.dimensions["time"].size == p["n_lead_times"]:
                        run_exist = True
                        if p["overwrite"]:
                            os.remove(outfn)
                            run_exist = False    
                    else:
                        os.remove(outfn)
                    
            if run_exist:
                print("Nowcast %s_nowcast already exists in %s" % (startdate.strftime("%Y%m%d%H%M"),path_to_nwc))
    
            else:
                countnwc += 1
                print("Computing the nowcast (%02d) ..." % countnwc)
                
                print("Starttime: %s" % startdate.strftime("%Y%m%d%H%M"))
                
                ## redirect stdout to log file
                logfn =  os.path.join(path_to_nwc, "%s_log.txt" % startdate.strftime("%Y%m%d%H%M")) 
                print("Log: %s" % logfn)
                orig_stdout = sys.stdout
                f = open(logfn, 'w')
                sys.stdout = f
                
                print("*******************")
                print("* %s *****" % startdate.strftime("%Y%m%d%H%M"))
                print("* Parameter set : *")
    #            pprint.pprint(p) 
                print("*******************")
                
                print("--- Start of the run : %s ---" % (datetime.datetime.now()))
                
                ## time
                t0 = time.time()
            
                # Read inputs
    #            print("Read the data...")
                
                ## find radar field filenames
                input_files = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern,
                                                  ds.fn_ext, ds.timestep, p["n_prvs_times"])
                
        
                ## read radar field files
                importer    = stp.io.get_method(ds.importer, type="importer")
                R, _, metadata = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
                metadata0 = metadata.copy()
                metadata0["shape"] = R.shape[1:]
                
                # Prepare input files
    #            print("Prepare the data...")
                
                ## if requested, make sure we work with a square domain
                reshaper = stp.utils.get_method(p["adjust_domain"])
                R, metadata = reshaper(R, metadata)
        
                ## if necessary, convert to rain rates [mm/h]    
                converter = stp.utils.get_method("mm/h")
                R, metadata = converter(R, metadata)
                
                ## threshold the data
                R[R < p["r_threshold"]] = 0.0
                metadata["threshold"] = p["r_threshold"]
                
                ## convert the data
                converter = stp.utils.get_method(p["unit"])
                R, metadata = converter(R, metadata)
                    
                ## transform the data
                transformer = stp.utils.get_method(p["transformation"])
                R, metadata = transformer(R, metadata)
                
                ## set NaN equal to zero
                R[~np.isfinite(R)] = metadata["zerovalue"]
                
                # Compute motion field
                oflow_method = stp.motion.get_method(p["oflow_method"])
                UV = oflow_method(R)
                
                #####
                # Perform the nowcast       
                #####
                
                ## initialize netcdf file
                incremental = "timestep" if p["nwc_method"].lower() == "steps" else None
                if catchments == True:
                    metadata_new = stp.utils.catchment_metadata_mpi(shapes, metadata0)
                    d = {}       
                    for n in range(0, len(catchment_filenames)):
                        d["exporter_{0}".format(n)] = stp.io.initialize_forecast_exporter_netcdf(outfn[n], startdate,
                                                      ds.timestep, p["n_lead_times"], metadata_new[n]["shape"], 
                                                      p["n_ens_members"], metadata_new[n], incremental=incremental)
                else:
                    exporter = stp.io.initialize_forecast_exporter_netcdf(outfn, startdate,
                                  ds.timestep, p["n_lead_times"], metadata0["shape"], 
                                  p["n_ens_members"], metadata0, incremental=incremental)
                
                ## start the nowcast
                nwc_method = stp.nowcasts.get_method(p["nwc_method"])
                R_fct = nwc_method(R, UV, p["n_lead_times"], p["n_ens_members"],
                                p["n_cascade_levels"], kmperpixel=metadata["xpixelsize"]/1000, 
                                timestep=ds.timestep, R_thr=metadata["threshold"], 
                                extrap_method=p["adv_method"], 
                                decomp_method=p["decomp_method"], 
                                bandpass_filter_method=p["bandpass_filter"], 
                                noise_method=p["noise_method"], 
                                noise_stddev_adj=p["noise_adjustment"],
                                vel_pert_method=p["vel_pert_method"],
                                ar_order=p["ar_order"],conditional=p["conditional"], 
                                probmatching_method=p["prob_matching"], 
                                mask_method=p["mask_method"], 
                                num_workers=p["num_workers"],
                                callback=export, 
                                return_output=False)
                
                ## save results, either per catchment or in total
                if catchments == True:
                    for n in range(0, len(catchment_filenames)):
                        key = list(d.keys())[n]
                        stp.io.close_forecast_file(d[key])
                else:
                    stp.io.close_forecast_file(exporter)
                R_fct = None
                
                # save log
                print("--- End of the run : %s ---" % (datetime.datetime.now()))
                print("--- Total time : %s seconds ---" % (time.time() - t0))
                sys.stdout = orig_stdout
                f.close()
                
            # next forecast
            startdate += datetime.timedelta(minutes = p["data"][2])



#    tr.print_diff()
#    scores.append(n)
    #### RETURN TO THE CORRECT DIRECTORY, JUST IN CASE SOMETHING WAS CHANGED...
  #  os.chdir('/u/imhof_rn/pysteps-master')

#### Wait here so we can collect all runs
#### Because we distributed the work evenly all processes should be here at approximately the same time
comm.Barrier()
#### Great, we're all here. Now let's gather the scores...
#### Collect values from all the processes in the main root
#scores = comm.gather(scores, root=0)

#logging.debug("Rank {} has scores {}".format(rank, scores))
  
end_time = time.time()

print('Total process took', (end_time - start_time)/3600.0, 'hours')  
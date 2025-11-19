def plot_exposure(fLogConsol: Path, _p: dict, ln_range: list =[0, 1000], _verbose: bool =False):
    """ Generate exposure plot for the given range of stimulus presentations 
        (as lines in the consolidated stimulus protocol file `fLogConsol`)
    """    
    # Read consolidated experiment protocol
    df = pd.read_csv(fLogConsol, on_bad_lines='warn', sep=';')

    # Recording field size and factor by which to increase recording field area
    # for intensity exposure traces
    rec_field_dxy_um = [95, 95]
    surr_factor = 1.0    

    # Determine the extent of the experiment area to create accumulator array
    df_valid = df[df['pos_xyz'].notna()].copy()
    df_valid['x'] = df_valid['pos_xyz'].apply(lambda v: float(v[1:-1].split(',')[0]))
    df_valid['y'] = df_valid['pos_xyz'].apply(lambda v: float(v[1:-1].split(',')[1]))

    # Set fixed bounds for the display area
    x_min = 500
    x_max = 2250
    y_min = -1250
    y_max = 250

    # Create accumulator array for the summary picture
    pix_size_um = _p["pix_size_um"]
    acc_width_px = int((x_max - x_min) / pix_size_um)
    acc_height_px = int((y_max - y_min) / pix_size_um)
    acc_image = np.zeros((acc_height_px, acc_width_px, _p["nCh"]), dtype=np.float64)
    if _verbose:
        print(f"Accumulator image size: {acc_width_px} x {acc_height_px} pixels")
        print(f"Spatial extent: X=[{x_min:.1f}, {x_max:.1f}] um, Y=[{y_min:.1f}, {y_max:.1f}] um")

    # Create variables to hold arrays for intensity-time plots
    int_mean = []
    int_cumul = []
    int_t_last_end_s = 0
    dt_fr_s = _p["dt_fr_s"]
    nCh = _p["nCh"]
    
    for index, row in df.iterrows():
        # Restrict to part of the experiment
        if index < ln_range[0] or index > ln_range[1]:
            continue

        # Get coordinates 
        if row["pos_xyz"] is not np.nan:
            s = row["pos_xyz"][1:-1].split(",")
            pos_xyz = [float(s[i]) for i in range(len(s))]
        else:
            pos_xyz = None    

        # Get other parameters
        fStimName = row["stimFileName"]
        fRecName = "" if row["dataFileName"] is np.nan else row["dataFileName"]
        t_abs_s = row["t_abs_s"]
        t_dur_s = row["t_dur_s"]
        
        if pos_xyz:
            # Consider only stimuli w/ position data
            x0, y0 = pos_xyz[0], pos_xyz[1]
            print(f"#{index:2d} {fStimName:20s} x,y={x0:.0f},{y0:.0f}")

            # Generate flattened stimulus movie and add it to the exposure picture
            if fStimName.upper() in ["DS"]:
                _mov = mov_DS
            elif fStimName.upper() in ["CHIRP"]:
                _mov = mov_Chirp
            elif fStimName.upper() in ["MOUSECAM_RIGHT"]:
                _mov = mov_MouseCamLeft
            else:
                continue
            mov_flat = stim_movies.flatten_movie(_mov, _p, _range_s=[0, t_dur_s])


            # Get intensity traces
            intens, intens_cumul = stim_movies.calc_intensity_trace(
                _mov, _p, 
                _field_size_um=[v *surr_factor for v in rec_field_dxy_um],
            )
            if len(int_mean) == 0:
                # First stimulus presentation
                int_mean = np.copy(intens)
                int_cumul = np.copy(intens_cumul)
                int_t_last_end_s = t_dur_s
            else:
                n = int((t_abs_s -int_t_last_end_s) *dt_fr_s)
                int_t_last_end_s += t_abs_s +int_t_last_end_s +t_dur_s
                if n <= 0:
                    print(f"WARNING: n={n} - why negative? ")
                else:
                    print("n=", n)
                    tmp = np.zeros((n,int_mean.shape[1]), dtype=np.float64)
                    int_mean = np.concatenate([int_mean, tmp])
                    int_cumul = np.concatenate([int_cumul, tmp])
                int_mean = np.concatenate([int_mean, intens])
                int_cumul = np.concatenate([int_cumul, intens_cumul])


            print("last end=", int_t_last_end_s, " stim start=", t_abs_s, " dur=", t_dur_s)
            print(int_mean.shape)

            
            # Ensure the movie has the correct number of channels
            mov_h, mov_w, mov_ch = mov_flat.shape
            if mov_ch != nCh:
                # Take only the first nCh channels
                mov_flat = mov_flat[:, :, :nCh]
            
            # Calculate position in accumulator array
            # Movie center is at (x0, y0), movie origin is at top-left
            mov_h, mov_w, _ = mov_flat.shape
            mov_center_x_um = x0
            mov_center_y_um = y0
            
            # Convert to pixel coordinates in accumulator
            # Note: Image coordinates have origin at top-left, so we need to flip y
            acc_center_x_px = int((mov_center_x_um - x_min) / pix_size_um)
            acc_center_y_px = int((mov_center_y_um - y_min) / pix_size_um)
            
            # Calculate bounds for placing the movie
            acc_x0 = acc_center_x_px - mov_w // 2
            acc_x1 = acc_x0 + mov_w
            acc_y0 = acc_center_y_px - mov_h // 2
            acc_y1 = acc_y0 + mov_h
            
            # Clip to accumulator bounds and add
            mov_x0 = max(0, -acc_x0)
            mov_x1 = mov_w - max(0, acc_x1 - acc_width_px)
            mov_y0 = max(0, -acc_y0)
            mov_y1 = mov_h - max(0, acc_y1 - acc_height_px)
            
            acc_x0_clip = max(0, acc_x0)
            acc_x1_clip = min(acc_width_px, acc_x1)
            acc_y0_clip = max(0, acc_y0)
            acc_y1_clip = min(acc_height_px, acc_y1)
            
            # Add the flattened movie to accumulator
            acc_image[acc_y0_clip:acc_y1_clip, acc_x0_clip:acc_x1_clip, :] += \
                mov_flat[mov_y0:mov_y1, mov_x0:mov_x1, :]

    # Normalize the accumulated image for display
    acc_image_norm = acc_image / (acc_image.max() + 1e-10)

    # Create figure for intensity plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = ['violet', 'green', 'blue']  # for channels 0, 1, 2
    time_s = np.arange(int_mean.shape[0]) *dt_fr_s

    # Mean intensity trace
    for ch in range(nCh):
        axes[0].plot(time_s, int_mean[:, ch], color=colors[ch], label=f'Ch {ch}')
    axes[0].set_ylabel('Mean Intensity')
    axes[0].set_title('Mean Intensity Over Time')
    axes[0].legend()
    axes[0].grid(True)

    # Cumulative intensity trace
    for ch in range(nCh):
        axes[1].plot(time_s, int_cumul[:, ch], color=colors[ch], label=f'Ch {ch}')
    axes[1].set_xlabel('t [s]')
    axes[1].set_ylabel('Cumulative Intensity')
    axes[1].set_title('Cumulative Intensity Over Time')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Create figure for plotting of exposure images
    fig, axes = plt.subplots(1, _p["nCh"], figsize=(12, 6))
    if nCh == 1:
        axes = [axes]

    colors = ['violet', 'green', 'blue']
    for ch in range(nCh):
        im = axes[ch].imshow(acc_image_norm[:, :, ch], cmap='gray', 
                            extent=[x_min, x_max, y_max, y_min], 
                            origin='upper', aspect='equal')
        axes[ch].set_xlabel('X [um]')
        axes[ch].set_ylabel('Y [um]')
        axes[ch].set_title(f'Channel {ch} ({colors[ch]})')
        axes[ch].set_xlim(x_min, x_max)
        axes[ch].set_ylim(y_min, y_max)
        
        # Draw gray rectangles centered at recording positions
        from matplotlib.patches import Rectangle
        rec_size = rec_field_dxy_um[0]
        for _, row_data in df_valid.iterrows():
            rect = Rectangle((row_data['x'] - rec_size/2, row_data['y'] - rec_size/2),
                            rec_size, rec_size,
                            linewidth=1, edgecolor='red', facecolor='none', zorder=9)
            axes[ch].add_patch(rect)
        
        # Add scatter points for recording positions
        axes[ch].scatter(df_valid['x'], df_valid['y'], c='red', 
                        s=20, marker='o', zorder=10, edgecolor='k', linewidth=0.5)
        
        #axes[ch].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
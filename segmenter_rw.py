# Use the RW algorithm to segment in 3D at one time point
# Make the segmentation slightly larger and extend it to all other time points
# Save the segmentated image using the hpc-predict-io classes

# ============================   
# import module and set paths
# ============================   
import numpy as np
from mr_io import FlowMRI, SegmentedFlowMRI
import matplotlib.pyplot as plt
import tkinter as tki
from PIL import Image, ImageTk
import utils
import rw3D
import rw4D

basepath_image = '/tmp/test.decrypt7/flownet/hpc_predict/v2/inference/2021-02-11_19-41-32_daint102'
# basepath_seg = '/tmp/test.decrypt7/segmenter/cnn_segmenter/hpc_predict/v1/inference/2021-02-11_19-41-32_daint102'
subnum = 3 # 1/2/3/4/5/6/7
flowmripath = basepath_image + '_volN' + str(subnum) + '/output/recon_volN' + str(subnum) + '_vn.mat.h5'
segmentedflowmripath = basepath_image + '_volN' + str(subnum) + '/output/recon_volN' + str(subnum) + '_vn_seg_rw.h5'

# ============================   
# Interactive segmentation tool for 4D MRI Flow Images
# ============================   
class MainWindow():
        
    # ============================   
    # Read the MRI image using the hpc-predict-io classes
    # Loading data (a FlowMRI object written by Flownet and in the filename given by inference_input)
    # ============================   
    flow_mri = FlowMRI.read_hdf5(flowmripath)
    print('shape of input intensity: ' + str(flow_mri.intensity.shape))
    print('shape of velocity mean: ' + str(flow_mri.velocity_mean.shape))
    # ============================   
    # conbine the intensity and velocity information into 1 array
    # ============================   
    flowMRI_image = np.concatenate([np.expand_dims(flow_mri.intensity, -1), flow_mri.velocity_mean], axis=-1)  
    # ============================
    # move the axes around so that we have [nz, nx, ny, nt, num_channels]
    # ============================
    # flowMRI_image = flowMRI_image.transpose([2, 0, 1, 3, 4])
    
    # ============================
    # normalize the arrays so that velocity and magnitude are in the range [0,1]
    # also this function uses the 95th percentile to normalize the data and clips any datapoint that is larger to 1.0 to get rid of outliers
    # skipping this for the manual segmentation...
    # ============================
    flowMRI_image = utils.normalize_arrays(flowMRI_image)
    
    # ============================
    # get array dimensions
    # ============================
    x_size, y_size, z_size, t_size, num_channels = flowMRI_image.shape
     
    # Set the initial t and z indices to the center of the respective axes
    img_timestep = round(t_size/2)
    img_slice = round(z_size/2)
    
    # this array determines what is visualized in the first window - either the magnitude image or the velocity magnitude
    img_array = flowMRI_image[..., 0]
    
    # create placeholders / initialize variables for the different arrays and lists that the GUI uses    
    
    # These lists will store the coordinates of the points marked by the user
    fg_list_2d = []
    bg_list_2d = []
    markers_3d = np.zeros((flowMRI_image.shape[0], flowMRI_image.shape[1], flowMRI_image.shape[2]))
    markers_4d = np.zeros((flowMRI_image.shape[0], flowMRI_image.shape[1], flowMRI_image.shape[2], flowMRI_image.shape[3]))
    
    # This array will contain the segmentation predicted by the algorithm
    rw_labels = np.zeros((flowMRI_image.shape[0], flowMRI_image.shape[1], flowMRI_image.shape[2], flowMRI_image.shape[3]))
    # This array will contain the segmentation to be visualized
    rw_labels_view = np.zeros((flowMRI_image.shape[0], flowMRI_image.shape[1], flowMRI_image.shape[2], flowMRI_image.shape[3]))
    
    # ---------------------------------------------------------------
    # Define the needed callbacks and functions for the GUI to run
    # ---------------------------------------------------------------
    def update_image(self):
        
        # decide what to show
        plt.figure()
        plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep], cmap='gray')
        plt.xticks([], []); plt.yticks([], [])
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.savefig('Tkimg.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        
        # plot and save the segmentation made so far        
        plt.figure()
        plt.imshow(self.img_array[:, :, self.img_slice, self.img_timestep], cmap='gray', alpha = 0.8)
        plt.imshow(self.rw_labels_view[:, :, self.img_slice, self.img_timestep], cmap='Reds', alpha = 0.3)
        plt.xticks([], []); plt.yticks([], [])
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.savefig('Tksegimg.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        
        # load the image and display on canvas
        pngimage = Image.open('.//Tkimg.png').resize(size=(self.x_size*3, self.y_size*3),resample = Image.BICUBIC)
        self.img =  ImageTk.PhotoImage(image = pngimage)
        self.canvas.create_image(0,0, anchor=tki.NW, image = self.img)
        
        pngsegimage = Image.open('.//Tksegimg.png').resize(size=(self.x_size*3, self.y_size*3),resample = Image.BICUBIC)
        self.segimg =  ImageTk.PhotoImage(image = pngsegimage)
        self.canvas2.create_image(0, 0, anchor=tki.NW, image = self.segimg)

        return
        
    # ---------------------------------------------------------------    
    # use slider1 to set the desired slice
    # ---------------------------------------------------------------    
    def update_z_axis(self):
        self.img_slice = self.slider1.get()
        self.update_image()
        return
        
    # ---------------------------------------------------------------    
    # use slider2 to set the desired timestep   
    # ---------------------------------------------------------------    
    def update_t_axis(self):
        self.img_timestep = self.slider2.get()
        self.update_image()
        return
    
    # ---------------------------------------------------------------    
    # loop to manage what to display (magnitude, overlap, markers, etc.)
    # ---------------------------------------------------------------    
    def display_mode(self):
          
        if self.button_display_image['text'] == "Velocity Magnitude":
            self.button_display_image.configure(text = "Intensity")
            self.img_array = utils.norm(self.flowMRI_image[..., 1],
                                        self.flowMRI_image[..., 2],
                                        self.flowMRI_image[..., 3])
            
        elif self.button_display_image['text'] == "Intensity":
            self.button_display_image.configure(text = "Velocity Magnitude")
            self.img_array = self.flowMRI_image[..., 0]
            
        self.update_image()
            
        return
        
    # ---------------------------------------------------------------    
    # ---------------------------------------------------------------    
    def mousecallback(self,event):
        x, y = event.x, event.y
        coord_tuple = (x,y)
        if x > 0 and y > 0 and x < self.x_size*3 and y < self.y_size*3:
            if self.v.get() == 1:
                if coord_tuple not in self.fg_list_2d:
                    self.canvas.create_oval(x, y, x+3, y+3, fill='green')
                    self.fg_list_2d.append(coord_tuple)
            elif self.v.get() == 2:
                if coord_tuple not in self.bg_list_2d:
                    self.canvas.create_oval(x, y, x+3, y+3, fill='red')
                    self.bg_list_2d.append(coord_tuple)
            else:
                return
    
    # ---------------------------------------------------------------  
    # ---------------------------------------------------------------  
    def scribble_draw(self):
        self.canvas.bind("<B1-Motion>", self.mousecallback)
        self.slider1.config(state = tki.DISABLED)
        self.slider2.config(state = tki.DISABLED)
        self.button_add_scribbles.config(state = tki.NORMAL)        
        return
    
    # ---------------------------------------------------------------  
    # ---------------------------------------------------------------  
    def resetvalues(self):
        self.v.set(0)
        self.button_scribble_fg.deselect()
        self.button_scribble_bg.deselect()
        self.slider1.config(state = tki.NORMAL)
        self.slider2.config(state = tki.NORMAL)
        self.canvas.delete("all")
        
        self.update_image()
        
        self.fg_list_2d = []
        self.bg_list_2d = []
        
        self.rw_labels = np.zeros((self.flowMRI_image.shape[0], self.flowMRI_image.shape[1], self.flowMRI_image.shape[2], self.flowMRI_image.shape[3]))
        self.rw_labels_view = np.zeros((self.flowMRI_image.shape[0], self.flowMRI_image.shape[1], self.flowMRI_image.shape[2], self.flowMRI_image.shape[3]))
        
        return
    
    # ---------------------------------------------------------------  
    # ---------------------------------------------------------------  
    def add_scribble(self):
        self.v.set(0)
        self.button_scribble_fg.deselect()
        self.button_scribble_bg.deselect()
        self.slider1.config(state = tki.NORMAL)
        self.slider2.config(state = tki.NORMAL)
        self.canvas.delete("all")
        self.update_image()
        self.add_markers_to_3d_list()        
        return

    # ---------------------------------------------------------------  
    # ---------------------------------------------------------------              
    def add_markers_to_3d_list(self):
        if self.fg_list_2d or self.bg_list_2d:
            self.markers_2d = np.zeros(self.flowMRI_image[:,:,0,0,0].shape)
            for t in self.fg_list_2d:
                self.markers_2d[round(t[1] / 3), round(t[0] / 3)] = 1
            for t in self.bg_list_2d:
                self.markers_2d[round(t[1] / 3), round(t[0] / 3)] = 2
                    
            self.markers_3d[:, :, self.img_slice] = self.markers_2d            
            self.fg_list_2d = []
            self.bg_list_2d = []
            
            print("---------- end of interaction....")
            for zz in range(self.z_size):
                tmp = len(np.nonzero(self.markers_3d[:, :, zz])[0])
                if (tmp != 0):
                    print(str(tmp) + 'markers in slice ' + str(zz))
            
        return
    
    # ---------------------------------------------------------------              
    # ---------------------------------------------------------------              
    def run_random_walker3D(self):
                
        alpha_a = 0.2
        beta_b = 0.4
        gamma_g = 1.0 - alpha_a - beta_b
        a ,b ,c = 200.0, 6.0, 500.0
    
        rw_labels3D = rw3D.random_walker(self.flowMRI_image[:, :, :, self.img_timestep, :],
                                         self.markers_3d,
                                         mode = 'cg_mg',
                                         return_full_prob = True,
                                         alpha = alpha_a,
                                         beta = beta_b,
                                         gamma = gamma_g,
                                         a = a,
                                         b = b,
                                         c = c)        

        # Extend the segmentation a bit, so that it includes the aorta at all time points
        eroded_labels3D = utils.erode_segmentation(np.round(rw_labels3D[0, :, :, :]))
        print(eroded_labels3D.shape)
        print(self.markers_4d.shape)
        
        # now, extend the 3D segmentation at this time instant to all time instances
        for tt in range(self.t_size):
            self.rw_labels[:, :, :, tt] = rw_labels3D[0, :, :, :] # the randow walker returns the prob. of the aorta in the first axis at index 0
            self.markers_4d[:, :, :, tt] = eroded_labels3D
            
        # push the predicted segmentation to the GUI
        self.update_image()
            
        return

    # ---------------------------------------------------------------   
    # The 3D segmentation should have been done satisfactorily at one time instance. 
    # This segmentation would then have been eroded and extended to all other time points.
    # This provides a band in between the foreground and the background, where the segmentation is done in 3D for each time point separately.           
    # ---------------------------------------------------------------                 
    def run_random_walker4D3D(self):

        alpha_a = 0.2
        beta_b = 0.4
        gamma_g = 1.0 - alpha_a - beta_b
        a, b, c = 200.0, 6.0, 500.0

        for tt in range(self.t_size):
            rw_labels3D = rw3D.random_walker(self.flowMRI_image[:, :, :, tt, :],
                                             self.markers_4d[:, :, :, tt],
                                             mode = 'cg_mg',
                                             return_full_prob = True,
                                             alpha = alpha_a,
                                             beta = beta_b,
                                             gamma = gamma_g,
                                             a = a,
                                             b = b,
                                             c = c)
        
            self.rw_labels[:, :, :, tt] = rw_labels3D[0, :, :, :]
                
        # push the predicted segmentation to the GUI
        self.update_image()
        
        return
    
    # ---------------------------------------------------------------   
    # The 3D segmentation should have been done satisfactorily at one time instance. 
    # This segmentation would then have been eroded and extended to all other time points.
    # This provides a band in between the foreground and the background, where the segmentation is done in 4D.   
    # (For some reason, this is causing NaNs in the predictions. Therefore, using run_random_walker4D3D for now.)        
    # ---------------------------------------------------------------                 
    def run_random_walker4D(self):

        alpha_a = 0.2
        beta_b = 0.4
        gamma_g = 1.0 - alpha_a - beta_b
        a, b, c = 200.0, 6.0, 500.0

        rw_labels4D = rw4D.random_walker(self.flowMRI_image,
                                         self.markers_4d,
                                         mode = 'cg_mg',
                                         return_full_prob = True,
                                         alpha = alpha_a,
                                         beta = beta_b,
                                         gamma = gamma_g,
                                         a = a,
                                         b = b,
                                         c = c)
        
        self.rw_labels = rw_labels4D[0, :, :, :, :]
                
        # push the predicted segmentation to the GUI
        self.update_image()
        
        return
    
    # ---------------------------------------------------------------  
    # ---------------------------------------------------------------  
    def toggle_segmentation(self):
        if self.button_display_segmentation['text'] == "View soft segmentation":
            self.button_display_segmentation.configure(text="View hard segmentation")
            self.rw_labels_view = self.rw_labels
            self.update_image()

        elif self.button_display_segmentation['text'] == "View hard segmentation":
            self.button_display_segmentation.configure(text="View eroded segmentation")
            self.rw_labels_view = np.round(self.rw_labels)
            self.update_image()
            
        elif self.button_display_segmentation['text'] == "View eroded segmentation":
            self.button_display_segmentation.configure(text="View soft segmentation")
            self.rw_labels_view = np.round(self.markers_4d)
            self.update_image()
            
        return
    
    # ---------------------------------------------------------------  
    # ---------------------------------------------------------------      
    def save_segmentation(self):

        # ============================
        # create an instance of the SegmentedFlowMRI class, with the image information from flow_mri as well as the predicted segmentation probabilities
        # ============================
        segmented_flow_mri = SegmentedFlowMRI(self.flow_mri.geometry,
                                              self.flow_mri.time,
                                              self.flow_mri.time_heart_cycle_period,
                                              self.flow_mri.intensity,
                                              self.flow_mri.velocity_mean,
                                              self.flow_mri.velocity_cov,
                                              self.rw_labels)

        # ============================
        # write SegmentedFlowMRI to file
        # ============================
        segmented_flow_mri.write_hdf5(segmentedflowmripath)
        # segmented_flow_mri = SegmentedFlowMRI.read_hdf5(segmentedflowmripath)
        
        return
        
    # ---------------------------------------------------------------  
    # Here all the elements of the GUI are defined (Buttons, Sliders, Canvas and Labels) with their sizes, values and callbacks (command)
    # ---------------------------------------------------------------      
    def __init__(self, main):
        
        # =========================================================    
        # Canvases
        # =========================================================    
        self.canvas = tki.Canvas(main,width=self.x_size*3, height=self.y_size*3, background='white')        
        self.canvas2 = tki.Canvas(main,width=self.x_size*3, height=self.y_size*3, background='white')
        
        self.canvas.grid(row=1,column=2,rowspan=7,sticky=tki.W)
        self.canvas2.grid(row=1,column=3,rowspan=7,sticky=tki.W)

        # =========================================================    
        # Sliders
        # =========================================================            
        self.slider1 = tki.Scale(main,from_=0,to=self.z_size-1,length=200,tickinterval=5,orient=tki.HORIZONTAL,label = "Z-Axis",command= lambda x: self.update_z_axis())
        self.slider1.set(round(self.z_size/2))        
        self.slider2 = tki.Scale(main, from_=0, to=self.t_size-1, length=200, tickinterval=5, orient=tki.HORIZONTAL,label = "T-Axis", command= lambda x: self.update_t_axis())
        self.slider2.set(round(self.t_size/2))
        
        self.slider1.grid(row=0,column =1,padx=5,pady=5)
        self.slider2.grid(row=1,column =1,padx=5,pady=5)
                
        # =========================================================
        # Buttons
        # =========================================================               
        self.v = tki.IntVar()

        # buttons to add user scribbles        
        self.button_scribble_fg = tki.Radiobutton(main, text="Scribble FG", variable=self.v, value=1, indicatoron=0, width=20, command = lambda: self.scribble_draw())
        self.button_scribble_bg = tki.Radiobutton(main, text="Scribble BG", variable=self.v, value=2, indicatoron=0, width=20, command = lambda: self.scribble_draw())
        self.button_add_scribbles = tki.Button(main, text='Add scribbles', width = 20, command = lambda: self.add_scribble())
        self.button_add_scribbles.config(state = tki.DISABLED)        
        
        # button to call the RW
        self.button_run_rw3D = tki.Button(main, text="Run RW 3D", width=20, command= lambda: self.run_random_walker3D())
        self.button_run_rw4D = tki.Button(main, text="Run RW 4D", width=20, command= lambda: self.run_random_walker4D3D())
        
        # misc buttons
        self.button_display_image = tki.Button(main, text="Intensity", width=20, command = lambda: self.display_mode())        
        self.button_display_segmentation = tki.Button(main, text="View hard segmentation", width = 20, command= lambda: self.toggle_segmentation())
        self.button_save = tki.Button(main, text="Save segmentation", width=20, command = lambda: self.save_segmentation())
        self.button_reset = tki.Button(main, text="Reset", width = 20, command= lambda: self.resetvalues()) # reset 2D
        self.button_quit = tki.Button(main, text='Quit', width=20, command=main.destroy)
        
        self.button_scribble_fg.grid(row=4, column=1, padx=5, pady=5)
        self.button_scribble_bg.grid(row=5, column=1, padx=5, pady=5)
        self.button_add_scribbles.grid(row=6, column=1, padx=5,pady=5)
        self.button_run_rw3D.grid(row=2, column=1, padx=5, pady=5)
        self.button_run_rw4D.grid(row=3, column=1, padx=5, pady=5)
        self.button_display_image.grid(row=8, column=2,padx=5, pady=5)
        self.button_display_segmentation.grid(row=8, column=4, padx=5, pady=5)
        self.button_save.grid(row=9, column=1, padx=5, pady=5)   
        self.button_reset.grid(row=100, column=2, padx=5, pady=5)
        self.button_quit.grid(row=100, column=1,padx=5, pady=5)
        
#----------------------------------------------------------------------------------------------------------
# TKinter main looop
#----------------------------------------------------------------------------------------------------------
root = tki.Tk()
root.title('Manual Segmentation GUI')
MainWindow(root)
root.mainloop()
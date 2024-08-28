import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import sys
import subprocess
import threading
from queue import Queue

from environment_functions import evolve_network
from environment_functions import render_game
from config_manipulator import get_config_file_path
from config_manipulator import restore_default_config
from config_manipulator import restore_all_default_configs


class MainWindow(tk.Tk):
    def __init__(self, width=1200, height=600, sidebar_width=400):
        super().__init__()

        # set basic geometry variables
        self.width = width
        self.height = height
        self.sidebar_width = sidebar_width

        # configure main window
        self.geometry(f'{self.width}x{self.height}')
        self.resizable(False, False)
        self.title('Training neuroevolution algorithms')

        # set an icon for the window
        icon = tk.PhotoImage(file='icon.png')
        self.iconphoto(False, icon)

        # initialize queue for communication between threads
        self.queue = Queue()

        # initialize program variables
        self.environment = tk.StringVar()
        self.algorithm = tk.StringVar()
        self.n_generations = tk.IntVar()
        self.use_multiprocessing = tk.BooleanVar()
        self.report_stats = tk.BooleanVar()

        # set default values for checkbutton variables
        self.use_multiprocessing.set(True)
        self.report_stats.set(True)

        # initialize widgets that will be used later in the program
        self.main_button = ttk.Button(
            self,
            text='Train',
            command=self.run_training
        )
        self.canvas = tk.Canvas(
            self,
            background='grey'
        )

        # configure style for window widgets
        style = ttk.Style(self)
        style.configure('TMenubutton', font=('Cooper Black', sidebar_width // 20))
        style.configure('TButton', font=('Cooper Black', sidebar_width // 28), justify='center')
        style.configure('TLabel', font=('Cooper Black', sidebar_width // 32))

        # initialize lists with values for dropdown menus
        self.available_environments = ['Acrobot', 'Cart Pole', 'Mountain Car', 'Lunar Lander']
        self.available_algorithms = ['NEAT', 'HyperNEAT', 'CMA-ES', 'Differential Evolution']

        # initialize and display all gui elements
        self._render_gui_elements()

        self.mainloop()

    def _render_gui_elements(self):

        # initialize list that will contain all sidebar elements
        sidebar_rows = []

        # initialize dropdown menus
        environment_dropdown = ttk.OptionMenu(
            self,
            self.environment,
            self.available_environments[0],
            *self.available_environments
        )
        algorithm_dropdown = ttk.OptionMenu(
            self,
            self.algorithm,
            self.available_algorithms[0],
            *self.available_algorithms
        )

        # initialize buttons that work with config files
        edit_config_button = ttk.Button(
            self,
            text='Edit config file',
            command=self.edit_config
        )
        restore_config_button = ttk.Button(
            self,
            text='Restore config file\nto default',
            command=self.restore_config
        )
        restore_all_button = ttk.Button(
            self,
            text='Restore all config files to default',
            command=restore_all_default_configs
        )

        # initialize spinbox to set the number of generations, restrict user input to numbers, initialize value to 10
        generation_spinbox_label = ttk.Label(
            self,
            text='Generations: ',
            anchor='e'
        )
        validate_cmd = (self.register(lambda value: value.isdigit()), "%P")
        generation_spinbox = ttk.Spinbox(
            self,
            textvariable=self.n_generations,
            from_=1,
            to=1000,
            increment=1,
            font=('Arial', self.sidebar_width // 20, 'bold'),
            justify='center',
            validate="key",
            validatecommand=validate_cmd
        )
        generation_spinbox.set(10)

        # initialize checkbuttons with corresponding text
        multiprocessing_checkbutton_label = ttk.Label(
            self,
            text='Use multiprocessing: ',
            anchor='e'
        )
        multiprocessing_checkbutton = ttk.Checkbutton(
            self,
            variable=self.use_multiprocessing
        )

        stats_checkbutton_label = ttk.Label(
            self,
            text='Report statistics: ',
            anchor='e'
        )
        stats_checkbutton = ttk.Checkbutton(
            self,
            variable=self.report_stats
        )

        # append sidebar rows in order
        sidebar_rows.append((environment_dropdown,))
        sidebar_rows.append((algorithm_dropdown,))
        sidebar_rows.append((edit_config_button, restore_config_button))
        sidebar_rows.append((restore_all_button,))
        sidebar_rows.append((generation_spinbox_label, generation_spinbox))
        sidebar_rows.append((multiprocessing_checkbutton_label, multiprocessing_checkbutton))
        sidebar_rows.append((stats_checkbutton_label, stats_checkbutton))

        # place the main "Train" / "Play Game" button at the bottom of the sidebar
        sidebar_rows.append((self.main_button,))

        # render all elements initialized above
        self._draw_sidebar(sidebar_rows)

        # place canvas to the right from the sidebar
        self.canvas.place(
            x=self.sidebar_width,
            y=0,
            width=self.width - self.sidebar_width,
            height=self.height
        )

    def _draw_sidebar(self, sidebar_rows):

        n_rows = len(sidebar_rows)
        for i in range(n_rows):
            n_columns = len(sidebar_rows[i])

            for j in range(n_columns):
                sidebar_rows[i][j].place(
                    x=self.sidebar_width // n_columns * j,
                    y=self.height // n_rows * i,
                    width=self.sidebar_width // n_columns,
                    height=self.height // n_rows
                )

    def _display_message(self, text):
        self.canvas.delete('all')

        self.canvas.create_text(
            (self.width - self.sidebar_width) // 2,
            self.height // 2,
            text=text,
            font=('Cooper Black', (self.width - self.sidebar_width) // 24),
            fill='dark red',
            justify='center'
        )

        self.canvas.update()

    def _render_game(self, network):
        DELAY_MILLISECONDS = 30  # delay 30ms gives approximately 30fps

        environment = render_game(self.environment.get(), network)
        rendered_frame = next(environment)

        frame_height = rendered_frame.shape[0]
        frame_width = rendered_frame.shape[1]
        while rendered_frame is not None:
            self.after(DELAY_MILLISECONDS)

            frame = Image.fromarray(rendered_frame)
            photo = ImageTk.PhotoImage(image=frame)

            self.canvas.delete('all')
            self.canvas.create_image(
                (self.width - self.sidebar_width - frame_width) // 2,
                (self.height - frame_height) // 2,
                anchor=tk.NW,
                image=photo
            )
            self.canvas.update()
            rendered_frame = next(environment)

    def play_game(self):
        self.main_button.config(
            state=tk.DISABLED
        )

        network = self.queue.get()
        self._render_game(network)

        self.main_button.config(
            command=self.run_training,
            text='Train',
            state=tk.NORMAL
        )

    def _wait_for_training(self):
        CHECKING_FREQUENCY = 500

        if self.queue.empty():
            self.after(CHECKING_FREQUENCY, self._wait_for_training)
        else:
            self._display_message('Algorithm trained.')
            self.main_button.config(
                command=self.play_game,
                text='Play Game',
                state=tk.NORMAL
            )

    def _train_algorithm(self):
        environment_name = self.environment.get()
        algorithm_name = self.algorithm.get()
        n_generations = self.n_generations.get()
        reporter = self.report_stats.get()
        multiprocessing = self.use_multiprocessing.get()

        evolved_network = evolve_network(environment_name, algorithm_name, n_generations, multiprocessing, reporter)
        self.queue.put(evolved_network)

    def run_training(self):
        computation_thread = threading.Thread(target=self._train_algorithm)
        computation_thread.start()

        self._display_message('Training...')
        self.main_button.config(state=tk.DISABLED)
        self._wait_for_training()

    def edit_config(self):
        config_path = get_config_file_path(self.environment.get(), self.algorithm.get())

        # open the config file in a text editor (command is based on the user's operating system)
        if sys.platform == 'win32':  # For Windows
            os.startfile(config_path)
        elif sys.platform == 'darwin':  # For macOS
            subprocess.run(['open', config_path])
        elif sys.platform == 'linux':  # For Linux
            subprocess.run(['xdg-open', config_path])
        else:
            raise NotImplementedError("Unsupported operating system")

    def restore_config(self):
        restore_default_config(self.environment.get(), self.algorithm.get())


if __name__ == '__main__':
    window = MainWindow()

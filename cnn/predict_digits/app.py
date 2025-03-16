import flet as ft
import flet.canvas as cv
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import os

class DrawingApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.state = {'x': None, 'y': None}
        self.shapes = []
        
        # Cargar modelo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'numeros_conv_ad_do.h5')
        self.model = tf.keras.models.load_model(model_path)
        
        # Inicializar componentes
        self.setup_ui()
        self.add_to_page()

    def setup_ui(self):
        self.page.title = "Number Recognition"
        self.page.vertical_alignment = ft.MainAxisAlignment.CENTER
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.page.bgcolor = ft.colors.BLACK12

        self.canvas = cv.Canvas(
            shapes=self.shapes,
            width=280,
            height=280,
        )

        self.gesture_detector = ft.GestureDetector(
            content=self.canvas,
            on_pan_start=self.pan_start,
            on_pan_update=self.pan_update,
            drag_interval=10,
        )

        self.result_text = ft.Text(value="", size=20)

        self.clean_button = ft.ElevatedButton(
            "Clean",
            on_click=self.clean,
            bgcolor=ft.colors.RED,
            color=ft.colors.WHITE
        )

        self.predict_button = ft.ElevatedButton(
            "Predict",
            on_click=self.predict_number,
            bgcolor=ft.colors.GREEN,
            color=ft.colors.WHITE
        )

    def add_to_page(self):
        self.page.add(
            ft.Column(
                controls=[
                    ft.Container(
                        content=self.gesture_detector,
                        border_radius=5,
                        bgcolor=ft.colors.WHITE,
                        width=280,
                        height=280,
                    ),
                    ft.Row(
                        controls=[self.clean_button, self.predict_button],
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=20
                    ),
                    self.result_text
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20
            )
        )

    def pan_start(self, e: ft.DragStartEvent):
        self.state['x'] = e.local_x
        self.state['y'] = e.local_y

    def pan_update(self, e: ft.DragUpdateEvent):
        line = cv.Line(
            x1=self.state['x'],
            y1=self.state['y'],
            x2=e.local_x,
            y2=e.local_y,
            paint=ft.Paint(
                stroke_width=10,
                color=ft.colors.BLACK,
                stroke_cap=ft.StrokeCap.ROUND,
                stroke_join=ft.StrokeJoin.ROUND
            )
        )
        self.shapes.append(line)
        self.state['x'] = e.local_x
        self.state['y'] = e.local_y
        self.canvas.update()

    def clean(self, e):
        self.shapes.clear()
        self.result_text.value = ""
        self.page.update()

    def predict_number(self, e):
        if not self.shapes:
            self.result_text.value = "Please draw a number first grrr!!!!"
            self.page.update()
            return

        # Crear y procesar la imagen
        canvas_image = Image.new('L', (280, 280), color=255)
        draw = ImageDraw.Draw(canvas_image)

        for shape in self.shapes:
            if isinstance(shape, cv.Line):
                draw.line(
                    [(shape.x1, shape.y1), (shape.x2, shape.y2)],
                    fill=0,
                    width=15
                )

        # Preprocesamiento
        canvas_image = canvas_image.resize((28, 28))
        img_array = np.array(canvas_image).astype('float32') / 255.0
        img_array = 1.0 - img_array
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predicción
        prediction = self.model.predict(img_array)
        predicted_number = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        self.result_text.value = f"Predicted: {predicted_number} ({confidence:.1f}%)"
        self.page.update()

def main(page: ft.Page):
    DrawingApp(page)

if __name__ == "__main__":
    ft.app(target=main)
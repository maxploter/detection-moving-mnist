import random
import math

import torch
import torchvision.transforms.functional as TF


class BaseTrajectory:
    def __init__(
        self, digit_label,
        affine_params,
        n,
        padding,
        initial_position,
        mnist_img,
        first_appearance_frame,
        canvas_width,
        canvas_height,
        **kwargs
    ):
        self.digit_label = digit_label
        self.affine_params = affine_params
        self.n = n
        self.padding = padding
        self.position = initial_position
        self.kwargs = kwargs
        self.mnist_img = mnist_img
        self.first_appearance_frame = first_appearance_frame
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # Set fixed initial values for the transformation
        self.translate = (
            random.randint(*self.affine_params.translate[0]),
            random.randint(*self.affine_params.translate[1]),
        )
        if self.translate[0] == 0:
            self.translate = (self.translate[0] + 1, self.translate[1])
        if self.translate[1] == 0:
            self.translate = (self.translate[0], self.translate[1] + 1)
        self.angle = random.uniform(*self.affine_params.angle)
        self.scale = random.uniform(*self.affine_params.scale)
        self.shear = random.uniform(*self.affine_params.shear)

    def transform(self, img, position):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __call__(self):

        # Place the digit in the center of a temporary canvas
        digit_canvas = torch.zeros((1, self.canvas_height, self.canvas_width), dtype=torch.float32)
        y_digit_min = digit_canvas.size(1) // 2 - self.mnist_img.size(1) // 2
        x_digit_min = digit_canvas.size(2) // 2 - self.mnist_img.size(2) // 2
        digit_canvas[:, y_digit_min:y_digit_min + self.mnist_img.size(1),
        x_digit_min:x_digit_min + self.mnist_img.size(2)] = self.mnist_img

        x = self.position[0]
        y = self.position[1]
        placed_img = TF.affine(digit_canvas, translate=[x, y], angle=0, scale=1, shear=[0])

        digit_bbox = self.bbox(placed_img)

        targets = {self.first_appearance_frame: {
            "frame": placed_img,
            "center_point": self.position,
            "bbox": digit_bbox,
        }}

        for t in range(self.first_appearance_frame+1, self.n):
            img, position = self.transform(digit_canvas, targets[t-1]['center_point'])

            targets[t] = {
                "frame": img,
                "center_point": position,
                "bbox": self.bbox(img),
            }
        return targets

    def bbox(self, img):
        """
        Calculate the bounding box of the digit in the image.
        Returns a tuple (x_min, y_min, width, height).
        """
        nonzero_mask = img > 0
        nonzero_indices = nonzero_mask.nonzero()
        if nonzero_indices.size(0) == 0:
            return None

        y_coords = nonzero_indices[:, 1]
        x_coords = nonzero_indices[:, 2]
        min_x = x_coords.min().item()
        max_x = x_coords.max().item()
        min_y = y_coords.min().item()
        max_y = y_coords.max().item()
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        return (min_x, min_y, width, height)


class SimpleLinearTrajectory(BaseTrajectory):
    def transform(self, img, position):
        img = TF.affine(
            img,
            angle=self.angle,
            translate=list(self.translate),
            scale=self.scale,
            shear=self.shear,
            **self.kwargs,
        )

        new_position = (
            position[0] + self.translate[0],
            position[1] + self.translate[1],
        )

        return img, new_position


class BouncingTrajectory(BaseTrajectory):
    def transform(self, img):
        new_position_x = self.position[0] + self.translate[0]
        new_position_y = self.position[1] + self.translate[1]

        # Check bounds
        if new_position_x <= -self.padding[0] or new_position_x >= self.padding[2]:
            self.translate = (-self.translate[0], self.translate[1])
        if new_position_y <= -self.padding[1] or new_position_y >= self.padding[3]:
            self.translate = (self.translate[0], -self.translate[1])

        self.position = (
            self.position[0] + self.translate[0],
            self.position[1] + self.translate[1],
        )

        img = TF.affine(
            img,
            angle=self.angle,
            translate=list(self.translate),
            scale=self.scale,
            shear=self.shear,
            **self.kwargs,
        )

        horizontal_direction, vertical_direction, size_change = self.describe_movement(
            self.translate, self.scale
        )
        transformation_caption = (
            f"The digit {self.digit_label} moves {horizontal_direction} by {abs(self.translate[0]):.1f} pixels and "
            f"{vertical_direction} by {abs(self.translate[1]):.1f} pixels, rotates by {self.angle:.1f} degrees, and {size_change}."
        )

        return img, transformation_caption


class OutOfBoundsTrajectory(BaseTrajectory):
    def __call__(self, img):
        # Add extra padding to handle out of bounds
        img = TF.pad(img, padding=[28, 28, 28, 28])

        sequence = [img]
        transformations = []
        for _ in range(self.n):
            img, caption = self.transform(sequence[-1])
            sequence.append(img)
            transformations.append(caption)

        # Remove the added extra padding
        for i, img in enumerate(sequence):
            sequence[i] = TF.center_crop(
                img,
                output_size=[
                    self.padding[1] + self.padding[3] + 28,
                    self.padding[0] + self.padding[2] + 28,
                ],
            )

        return sequence, transformations

    def transform(self, img):
        expanded_padding = [p + 28 for p in self.padding]

        new_position_x = self.position[0] + self.translate[0]
        new_position_y = self.position[1] + self.translate[1]

        # Check bounds
        if (
            new_position_x < -expanded_padding[0]
            or new_position_x > expanded_padding[2]
        ):
            self.translate = (-self.translate[0], self.translate[1])
        if (
            new_position_y < -expanded_padding[1]
            or new_position_y > expanded_padding[3]
        ):
            self.translate = (self.translate[0], -self.translate[1])

        self.position = (
            self.position[0] + self.translate[0],
            self.position[1] + self.translate[1],
        )

        img = TF.affine(
            img,
            angle=self.angle,
            translate=list(self.translate),
            scale=self.scale,
            shear=self.shear,
            **self.kwargs,
        )

        horizontal_direction, vertical_direction, size_change = self.describe_movement(
            self.translate, self.scale
        )

        # Check the actual bounds to create captions
        out_of_bounds_x = (
            self.position[0] < -self.padding[0] or self.position[0] > self.padding[2]
        )
        out_of_bounds_y = (
            self.position[1] < -self.padding[1] or self.position[1] > self.padding[3]
        )
        if out_of_bounds_x or out_of_bounds_y:
            transformation_caption = (
                f"The digit {self.digit_label} moves out of frame and disappears."
            )
        else:
            transformation_caption = (
                f"The digit {self.digit_label} moves {horizontal_direction} by {abs(self.translate[0]):.1f} pixels and "
                f"{vertical_direction} by {abs(self.translate[1]):.1f} pixels, rotates by {self.angle:.1f} degrees, and {size_change}."
            )

        return img, transformation_caption


class RandomTrajectory(BaseTrajectory):
    def transform(self, img):
        # Get random values for each transform
        angle = random.uniform(*self.affine_params.angle)
        translate = (
            random.randint(*self.affine_params.translate[0]),
            random.randint(*self.affine_params.translate[1]),
        )
        scale = random.uniform(*self.affine_params.scale)
        shear = random.uniform(*self.affine_params.shear)

        new_position_x = self.position[0] + translate[0]
        new_position_y = self.position[1] + translate[1]

        # Check bounds
        if new_position_x <= -self.padding[0] or new_position_x >= self.padding[2]:
            translate = (-translate[0], translate[1])
        if new_position_y <= -self.padding[1] or new_position_y >= self.padding[3]:
            translate = (translate[0], -translate[1])

        self.position = (
            self.position[0] + translate[0],
            self.position[1] + translate[1],
        )

        img = TF.affine(
            img,
            angle=angle,
            translate=list(translate),
            scale=scale,
            shear=shear,
            **self.kwargs,
        )

        horizontal_direction, vertical_direction, size_change = self.describe_movement(
            translate, scale
        )
        transformation_caption = (
            f"The digit {self.digit_label} moves {horizontal_direction} by {abs(translate[0]):.1f} pixels and "
            f"{vertical_direction} by {abs(translate[1]):.1f} pixels, rotates by {angle:.1f} degrees, and {size_change}."
        )

        return img, transformation_caption


class NonLinearTrajectory(BaseTrajectory):
    def __init__(
        self,
        digit_label,
        affine_params,
        n,
        padding,
        initial_position,
        mnist_img,
        first_appearance_frame,
        canvas_width,
        canvas_height,
        path_type="sine",
        amplitude=10,
        frequency=0.1,
        **kwargs
    ):
        super().__init__(
            digit_label, affine_params, n, padding, initial_position,
            mnist_img,
            first_appearance_frame,
            canvas_width,
            canvas_height,
            **kwargs)
        self.first_appearance_frame = first_appearance_frame
        self.path_type = path_type
        self.amplitude = amplitude
        self.frequency = frequency
        self.t = 0  # Time parameter for trajectory

    def transform(self, img, position):

        # Base movement
        base_x = position[0] #+ self.translate[0]
        base_y = position[1] #+ self.translate[1]

        # Apply non-linear component
        if self.path_type == "sine":
            offset_y = self.amplitude * math.sin(self.frequency * self.t)
            new_position = (base_x, base_y + offset_y)
        elif self.path_type == "circle":
            offset_x = self.amplitude * math.cos(self.frequency * self.t)
            offset_y = self.amplitude * math.sin(self.frequency * self.t)
            new_position = (base_x + offset_x, base_y + offset_y)
        elif self.path_type == "spiral":
            growing_amplitude = self.amplitude * (1 + 0.05 * self.t)
            offset_x = growing_amplitude * math.cos(self.frequency * self.t)
            offset_y = growing_amplitude * math.sin(self.frequency * self.t)
            new_position = (base_x + offset_x, base_y + offset_y)
        else:
            new_position = (base_x, base_y)

        # Use affine transformation to move the digit to the correct position
        img = TF.affine(
            img,
            angle=self.angle,
            translate=new_position,  # x, y are already relative to center
            scale=self.scale,
            shear=self.shear,
            **self.kwargs,
        )

        self.t += 1
        return img, new_position

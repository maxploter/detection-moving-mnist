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
        **kwargs
    ):
        super().__init__(
            digit_label, affine_params, n, padding, initial_position,
            mnist_img,
            first_appearance_frame,
            canvas_width,
            canvas_height,
            **kwargs)

        self.path_type = kwargs['path_type']
        self.t = 0  # Time parameter for trajectory

        # Store common parameters
        self.params = kwargs.copy()

    def transform(self, img, position):
        # Base position
        base_x = position[0]
        base_y = position[1]
        offset_x = 0
        offset_y = 0

        # Extract common parameters
        amplitude = self.params['amplitude']
        frequency = self.params['frequency']

        # Apply trajectory specific transformation
        if self.path_type == "sine":
            phase = self.params['phase']
            direction = self.params['direction']

            if direction == "vertical":
                offset_y = amplitude * math.sin(frequency * self.t + phase)
            elif direction == "horizontal":
                offset_x = amplitude * math.sin(frequency * self.t + phase)
            else:  # diagonal
                offset_x = amplitude * 0.7071 * math.sin(frequency * self.t + phase)
                offset_y = amplitude * 0.7071 * math.sin(frequency * self.t + phase)

        elif self.path_type == "circle":
            offset_x = amplitude * math.cos(frequency * self.t)
            offset_y = amplitude * math.sin(frequency * self.t)

            # Apply center offset if specified
            center_offset = self.params['center_offset']
            offset_x += center_offset[0]
            offset_y += center_offset[1]

        elif self.path_type == "spiral":
            growing_amplitude = amplitude * (1 + 0.05 * self.t)
            offset_x = growing_amplitude * math.cos(frequency * self.t)
            offset_y = growing_amplitude * math.sin(frequency * self.t)

        elif self.path_type == "polynomial":
            # Extract polynomial coefficients
            poly_coeffs = self.params['poly_coeffs']
            poly_scale = poly_coeffs.get('poly_scale', 1.0)

            # Calculate polynomial trajectories for both x and y
            poly_coeffs_x = poly_coeffs.get('x', [0, 1, 0])
            poly_coeffs_y = poly_coeffs.get('y', [0, 1, 0])

            offset_x = sum(coeff * (self.t ** i) for i, coeff in enumerate(poly_coeffs_x))
            offset_y = sum(coeff * (self.t ** i) for i, coeff in enumerate(poly_coeffs_y))

            # Apply scaling factor
            offset_x *= poly_scale
            offset_y *= poly_scale

        elif self.path_type == "figure8":
            scale_x = self.params['scale_x']
            scale_y = self.params['scale_y']
            angular_velocity = self.params['angular_velocity']
            rotation = self.params['rotation']

            # Figure 8 parametric equations
            raw_x = scale_x * math.sin(angular_velocity * self.t)
            raw_y = scale_y * math.sin(2 * angular_velocity * self.t)

            # Apply rotation if specified
            offset_x = raw_x * math.cos(rotation) - raw_y * math.sin(rotation)
            offset_y = raw_x * math.sin(rotation) + raw_y * math.cos(rotation)

        elif self.path_type == "elliptical":
            semi_major = self.params['semi_major']
            semi_minor = self.params['semi_minor']
            angular_velocity = self.params['angular_velocity']
            rotation = self.params['rotation']
            center_offset = self.params['center_offset']

            # Elliptical parametric equations
            raw_x = semi_major * math.cos(angular_velocity * self.t)
            raw_y = semi_minor * math.sin(angular_velocity * self.t)

            # Apply rotation
            offset_x = raw_x * math.cos(rotation) - raw_y * math.sin(rotation) + center_offset[0]
            offset_y = raw_x * math.sin(rotation) + raw_y * math.cos(rotation) + center_offset[1]

        elif self.path_type == "cubic":
            coefficients = self.params['coefficients']
            direction = self.params['direction']
            scale = self.params['scale']

            t = self.t * scale
            cubic_value = (coefficients['a'] * (t**3) +
                         coefficients['b'] * (t**2) +
                         coefficients['c'] * t +
                         coefficients['d'])

            if direction == "x":
                offset_x = cubic_value
            elif direction == "y":
                offset_y = cubic_value
            else:  # both
                offset_x = cubic_value * 0.7071
                offset_y = cubic_value * 0.7071

        elif self.path_type == "zigzag":
            segment_length = self.params['segment_length']
            angle = self.params['angle']
            direction = self.params['direction']
            randomness = self.params['randomness']

            # Basic zigzag pattern
            segment = (self.t / segment_length) % 2
            zigzag = abs(segment - 1) * 2 - 1  # Maps to [-1, 1]

            # Add randomness if specified
            if randomness > 0:
                zigzag += random.uniform(-randomness, randomness)

            if direction == "horizontal":
                offset_x = self.t
                offset_y = zigzag * amplitude
            elif direction == "vertical":
                offset_x = zigzag * amplitude
                offset_y = self.t
            else:  # diagonal
                offset_x = self.t * 0.7071
                offset_y = zigzag * amplitude * 0.7071

        elif self.path_type == "exponential":
            base = self.params['base']
            scale = self.params['scale']
            direction = self.params['direction']
            decay_factor = self.params['decay_factor']

            # Calculate exponential value with optional decay
            exp_value = scale * (base ** (self.t * (decay_factor ** self.t)))

            if direction == "right":
                offset_x = exp_value
            elif direction == "left":
                offset_x = -exp_value
            elif direction == "up":
                offset_y = -exp_value
            else:  # down
                offset_y = exp_value

        elif self.path_type == "hyperbolic":
            scale_x = self.params['scale_x']
            scale_y = self.params['scale_y']
            shift = self.params['shift']
            orientation = self.params['orientation']
            branch = self.params['branch']

            # Adjusted time value to avoid singularity
            t_adj = self.t + shift

            if orientation == "x":
                if branch == "positive" or (branch == "both" and self.t % 2 == 0):
                    offset_x = scale_x / t_adj
                else:
                    offset_x = -scale_x / t_adj
                offset_y = self.t
            else:  # y-orientation
                offset_x = self.t
                if branch == "positive" or (branch == "both" and self.t % 2 == 0):
                    offset_y = scale_y / t_adj
                else:
                    offset_y = -scale_y / t_adj

        # Calculate new position
        new_position = (base_x + offset_x + self.translate[0], base_y + offset_y + self.translate[1])

        # Use affine transformation to place the digit at the new position
        img = TF.affine(
            img,
            angle=self.angle,
            translate=new_position,
            scale=self.scale,
            shear=self.shear,
            fill=0
        )

        self.t += 1
        return img, new_position

import random

import torchvision.transforms.functional as TF


class BaseTrajectory:
    def __init__(
        self, digit_label, affine_params, n, padding, initial_position, **kwargs
    ):
        self.digit_label = digit_label
        self.affine_params = affine_params
        self.n = n
        self.padding = padding
        self.position = initial_position
        self.kwargs = kwargs

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

    def __call__(self, img):
        sequence = [img]
        center_points = [self.position]
        for _ in range(self.n):
            img, position = self.transform(sequence[-1], center_points[-1])
            sequence.append(img)
            center_points.append(position)
        return sequence, center_points


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
        first_appearance_frame=0,
        path_type="sine",
        amplitude=10,
        frequency=0.1,
        **kwargs
    ):
        super().__init__(digit_label, affine_params, n, padding, initial_position, **kwargs)
        self.first_appearance_frame = first_appearance_frame
        self.path_type = path_type
        self.amplitude = amplitude
        self.frequency = frequency
        self.t = 0  # Time parameter for trajectory

    def transform(self, img, position):
        import math

        # For frames before appearance, keep position off-screen
        if self.t < self.first_appearance_frame:
            self.t += 1
            # Return off-screen position
            return img, (-1000, -1000)

        # Base movement
        base_x = position[0] + self.translate[0]
        base_y = position[1] + self.translate[1]

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

        # Calculate relative translation for this frame
        relative_translation = [new_position[0] - position[0], new_position[1] - position[1]]

        img = TF.affine(
            img,
            angle=self.angle,
            translate=relative_translation,
            scale=self.scale,
            shear=self.shear,
            **self.kwargs,
        )

        self.t += 1
        return img, new_position

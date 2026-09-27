//! Geometry information

/// A bounding box describes geometry in which an Octree lives.
pub struct PhysicalBox {
    coords: [f64; 6],
}

impl PhysicalBox {
    /// Create a new bounding box.
    ///
    /// The coordinates are given by `[xmin, ymin, zmin, xmax, ymax, zmax]`.
    /// # Parameters
    ///
    /// - `coords`: `[xmin, ymin, zmin, xmax, ymax, zmax]` bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use nd_octree::PhysicalBox;
    /// let box_ = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(box_.coordinates()[3], 1.0);
    /// ```
    pub fn new(coords: [f64; 6]) -> Self {
        Self { coords }
    }

    /// Return coordinates
    /// # Examples
    ///
    /// ```
    /// use nd_octree::PhysicalBox;
    /// let box_ = PhysicalBox::new([0.0; 6]);
    /// assert_eq!(box_.coordinates(), [0.0; 6]);
    /// ```
    pub fn coordinates(&self) -> [f64; 6] {
        self.coords
    }

    /// Map a point from the reference box [0, 1]^3 to the bounding box.
    /// # Parameters
    ///
    /// - `point`: Three Cartesian coordinates in the source coordinate system.
    ///
    /// # Examples
    ///
    /// ```
    /// use nd_octree::PhysicalBox;
    /// let box_ = PhysicalBox::new([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]);
    /// assert_eq!(box_.reference_to_physical([0.5; 3]), [1.0; 3]);
    /// ```
    pub fn reference_to_physical(&self, point: [f64; 3]) -> [f64; 3] {
        let [xmin, ymin, zmin, xmax, ymax, zmax] = self.coords;

        [
            xmin + (xmax - xmin) * point[0],
            ymin + (ymax - ymin) * point[1],
            zmin + (zmax - zmin) * point[2],
        ]
    }

    /// Map a point from the physical domain to the reference box.
    /// # Parameters
    ///
    /// - `point`: Three Cartesian coordinates in the source coordinate system.
    ///
    /// # Examples
    ///
    /// ```
    /// use nd_octree::PhysicalBox;
    /// let box_ = PhysicalBox::new([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]);
    /// assert_eq!(box_.physical_to_reference([1.0; 3]), [0.5; 3]);
    /// ```
    pub fn physical_to_reference(&self, point: [f64; 3]) -> [f64; 3] {
        let [xmin, ymin, zmin, xmax, ymax, zmax] = self.coords;

        [
            (point[0] - xmin) / (xmax - xmin),
            (point[1] - ymin) / (ymax - ymin),
            (point[2] - zmin) / (zmax - zmin),
        ]
    }

    /// Return an ordered list of corners of the box.
    ///
    /// The ordering of the corners on the unit cube is
    /// [0, 0, 0]
    /// [1, 0, 0]
    /// [1, 1, 0]
    /// [0, 1, 0]
    /// [0, 0, 1]
    /// [1, 0, 1]
    /// [1, 1, 1]
    /// [0, 1, 1]
    /// # Examples
    ///
    /// ```
    /// use nd_octree::PhysicalBox;
    /// let box_ = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(box_.corners().len(), 8);
    /// ```
    pub fn corners(&self) -> [[f64; 3]; 8] {
        let reference_points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];

        [
            self.reference_to_physical(reference_points[0]),
            self.reference_to_physical(reference_points[1]),
            self.reference_to_physical(reference_points[2]),
            self.reference_to_physical(reference_points[3]),
            self.reference_to_physical(reference_points[4]),
            self.reference_to_physical(reference_points[5]),
            self.reference_to_physical(reference_points[6]),
            self.reference_to_physical(reference_points[7]),
        ]
    }
}

impl std::fmt::Display for PhysicalBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [xmin, ymin, zmin, xmax, ymax, zmax] = self.coords;

        write!(
            f,
            "(xmin: {}, ymin: {}, zmin: {}, xmax: {}, ymax: {}, zmax: {})",
            xmin, ymin, zmin, xmax, ymax, zmax
        )
    }
}

#[cfg(test)]
mod test {
    use super::PhysicalBox;

    #[test]
    fn test_reference_physical_round_trip_and_corner_order() {
        let bounding_box = PhysicalBox::new([-2.5, 3.0, -7.0, 5.5, 11.0, 1.0]);
        let reference_points = [[0.0, 0.0, 0.0], [0.125, 0.5, 0.875], [1.0, 1.0, 1.0]];

        for reference in reference_points {
            let physical = bounding_box.reference_to_physical(reference);
            assert_eq!(bounding_box.physical_to_reference(physical), reference);
        }

        assert_eq!(
            bounding_box.corners(),
            [
                [-2.5, 3.0, -7.0],
                [5.5, 3.0, -7.0],
                [5.5, 11.0, -7.0],
                [-2.5, 11.0, -7.0],
                [-2.5, 3.0, 1.0],
                [5.5, 3.0, 1.0],
                [5.5, 11.0, 1.0],
                [-2.5, 11.0, 1.0],
            ]
        );
    }
}

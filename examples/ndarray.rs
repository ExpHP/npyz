mod read_example {
    // Example of parsing to an array with fixed NDIM.
    fn to_array_3<T>(data: Vec<T>, shape: Vec<u64>, order: npyz::Order) -> ndarray::Array3<T> {
        use ndarray::ShapeBuilder;

        let shape = match shape[..] {
            [i1, i2, i3] => [i1 as usize, i2 as usize, i3 as usize],
            _ => panic!("expected 3D array"),
        };
        let true_shape = shape.set_f(order == npyz::Order::Fortran);

        ndarray::Array3::from_shape_vec(true_shape, data)
            .unwrap_or_else(|e| panic!("shape error: {}", e))
    }

    // Example of parsing to an array with dynamic NDIM.
    fn to_array_d<T>(data: Vec<T>, shape: Vec<u64>, order: npyz::Order) -> ndarray::ArrayD<T> {
        use ndarray::ShapeBuilder;

        let shape = shape.into_iter().map(|x| x as usize).collect::<Vec<_>>();
        let true_shape = shape.set_f(order == npyz::Order::Fortran);

        ndarray::ArrayD::from_shape_vec(true_shape, data)
            .unwrap_or_else(|e| panic!("shape error: {}", e))
    }

    pub fn main() -> std::io::Result<()> {
        let bytes = std::fs::read("test-data/c-order.npy")?;
        let reader = npyz::NpyFile::new(&bytes[..])?;
        let shape = reader.shape().to_vec();
        let order = reader.order();
        let data = reader.into_vec::<i64>()?;

        println!("{:?}", to_array_3(data.clone(), shape.clone(), order));
        println!("{:?}", to_array_d(data.clone(), shape.clone(), order));
        Ok(())
    }

    #[test]
    fn read_c_order() {
        let bytes = std::fs::read("test-data/c-order.npy").unwrap();
        let reader = npyz::NpyFile::new(&bytes[..]).unwrap();
        let order = reader.order();
        let shape = reader.shape().to_vec();
        let data = reader.into_vec::<i64>().unwrap();

        let arr3 = to_array_3(data.clone(), shape.clone(), order);
        let arrd = to_array_d(data.clone(), shape.clone(), order);

        // array looks like:
        //   1,1,1,1, 2,2,2,2, 3,3,3,3,
        //   4,4,4,4, 5,5,5,5, 6,6,6,6,
        assert_eq!(arr3.ndim(), 3);
        assert_eq!(arr3[[0, 2, 1]], 3);
        assert_eq!(arr3[[0, 1, 1]], 2);
        assert_eq!(arr3[[1, 2, 2]], 6);
        assert_eq!(arrd.ndim(), 3);
        assert_eq!(arrd[[0, 2, 1]], 3);
        assert_eq!(arrd[[0, 1, 1]], 2);
        assert_eq!(arrd[[1, 2, 2]], 6);
    }

    #[test]
    fn read_f_order() {
        let bytes = std::fs::read("test-data/f-order.npy").unwrap();
        let reader = npyz::NpyFile::new(&bytes[..]).unwrap();
        let order = reader.order();
        let shape = reader.shape().to_vec();
        let data = reader.into_vec::<i64>().unwrap();

        let arr3 = to_array_3(data.clone(), shape.clone(), order);
        let arrd = to_array_d(data.clone(), shape.clone(), order);

        // The resulting arrays should be the same as for c-order when using
        // multidimensional indices, even though internally the data is in a
        // completely different order.
        assert_eq!(arr3.ndim(), 3);
        assert_eq!(arr3[[0, 2, 1]], 3);
        assert_eq!(arr3[[0, 1, 1]], 2);
        assert_eq!(arr3[[1, 2, 2]], 6);
        assert_eq!(arrd.ndim(), 3);
        assert_eq!(arrd[[0, 2, 1]], 3);
        assert_eq!(arrd[[0, 1, 1]], 2);
        assert_eq!(arrd[[1, 2, 2]], 6);
    }
}

mod write_example {
    use std::fs::File;
    use std::io;

    use ndarray::Array;
    use npyz::WriterBuilder;

    // Example of writing an array with unknown shape.  The output is always C-order.
    fn write_array<T, S, D>(
        writer: impl io::Write,
        array: &ndarray::ArrayBase<S, D>,
    ) -> io::Result<()>
    where
        T: Clone + npyz::AutoSerialize,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let shape = array.shape().iter().map(|&x| x as u64).collect::<Vec<_>>();
        let c_order_items = array.iter();

        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&shape)
            .writer(writer)
            .begin_nd()?;
        writer.extend(c_order_items)?;
        writer.finish()
    }

    pub fn main() -> io::Result<()> {
        let array = Array::from_shape_fn((6, 7, 8), |(i, j, k)| {
            100 * i as i32 + 10 * j as i32 + k as i32
        });
        // even weirdly-ordered axes and non-contiguous arrays are fine
        let view = array.view(); // shape (6, 7, 8), C-order
        let view = view.reversed_axes(); // shape (8, 7, 6), fortran order
        let view = view.slice(ndarray::s![.., .., ..;2]); // shape (8, 7, 3), non-contiguous
        assert_eq!(view.shape(), &[8, 7, 3]);

        let mut file = io::BufWriter::new(File::create("examples/output/ndarray.npy")?);
        write_array(&mut file, &view)
    }

    #[test]
    fn test() -> io::Result<()> {
        let array = Array::from_shape_fn((6, 7, 8), |(i, j, k)| {
            100 * i as i32 + 10 * j as i32 + k as i32
        });
        let view = array.view();
        let view = view.reversed_axes();
        let view = view.slice(ndarray::s![.., .., ..;2]);
        assert_eq!(view.shape(), &[8, 7, 3]);

        let mut bytes = vec![];
        write_array(&mut bytes, &view)?;

        let npy = npyz::NpyFile::new(&bytes[..])?;
        assert_eq!(npy.order(), npyz::Order::C);
        assert_eq!(npy.shape(), &[8, 7, 3]);
        // check that the items were properly converted into C order
        let mut reader = npy.data::<i32>().unwrap();
        assert_eq!(reader.next().unwrap().unwrap(), 0);
        assert_eq!(reader.next().unwrap().unwrap(), 200);
        assert_eq!(reader.next().unwrap().unwrap(), 400);
        assert_eq!(reader.next().unwrap().unwrap(), 10);
        Ok(())
    }
}

fn main() {
    read_example::main().unwrap();
    write_example::main().unwrap();
}

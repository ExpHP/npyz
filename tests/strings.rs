use std::io::{self, Cursor};
use std::fs::File;
use npyz::WriterBuilder;

#[test]
fn unicode_files() {
    // Files created by:
    //
    // ```python
    // np.save(
    //     "test-data/unicode/ok.npy",
    //     np.array(["αβout"], dtype='<U8'),
    // )
    // np.save(
    //     "test-data/unicode/surrogate.npy",
    //     np.array(["\uD805"], dtype='<U1'),
    // )
    // # This is the surrogate pair for 𝄞 (U+1D11E) 'MUSICAL SYMBOL G CLEF',
    // # but since this is UTF-32 and not UTF-16, it is still not valid.
    // np.save(
    //     "test-data/unicode/surrogate-pair.npy",
    //     np.array(["\uD834\uDD1E"], dtype='<U2'),
    // )
    // ```

    fn read_file<T: npyz::Deserialize>(path: &str) -> io::Result<Vec<T>> {
        let file = File::open(path).unwrap_or_else(|e| panic!("{}: {}", path, e));
        let reader = npyz::NpyFile::new(file).unwrap();
        reader.into_vec::<T>()
    }

    assert_eq!(
        read_file::<String>("test-data/unicode/ok.npy").unwrap(),
        vec!["αβout".to_string()],
    );
    assert!(read_file::<String>("test-data/unicode/surrogate.npy").is_err());
    assert!(read_file::<String>("test-data/unicode/surrogate-pair.npy").is_err());

    assert_eq!(
        read_file::<Vec<char>>("test-data/unicode/ok.npy").unwrap(),
        vec!["αβout".chars().collect::<Vec<_>>()],
    );
    assert!(read_file::<Vec<char>>("test-data/unicode/surrogate.npy").is_err());
    assert!(read_file::<Vec<char>>("test-data/unicode/surrogate-pair.npy").is_err());

    assert_eq!(
        read_file::<Vec<u32>>("test-data/unicode/ok.npy").unwrap(),
        vec!["αβout".chars().map(|x| x as u32).collect::<Vec<_>>()],
    );
    assert_eq!(
        read_file::<Vec<u32>>("test-data/unicode/surrogate.npy").unwrap(),
        vec![vec![0xD805]],
    );
    assert_eq!(
        read_file::<Vec<u32>>("test-data/unicode/surrogate-pair.npy").unwrap(),
        vec![vec![0xD834, 0xDD1E]],
    );
}

#[test]
fn writing_strings() {
    let strings = vec![
        "abc".to_string(),
        "αβout".to_string(),
        "\u{1D11E}".to_string(),
    ];

    let utf32_strings: Vec<Vec<char>> = strings.iter().map(|str| str.chars().collect()).collect();

    fn check_writing<T: npyz::Serialize>(
        strings_to_write: &[T],
        expected_utf32s: &[Vec<char>],
    ) {
        let max_len = expected_utf32s.iter().map(|utf32| utf32.len()).max().unwrap();
        let dtype = npyz::DType::new_scalar(format!("<U{}", max_len).parse().unwrap());

        let mut buffer = Cursor::new(vec![]);
        let mut npy_writer = npyz::WriteOptions::new().dtype(dtype).writer(&mut buffer).begin_1d().unwrap();
        npy_writer.extend(strings_to_write).unwrap();
        npy_writer.finish().unwrap();

        let buffer = buffer.into_inner();
        let reader = npyz::NpyFile::new(&buffer[..]).unwrap();
        let read_utf32s = reader.into_vec::<Vec<char>>().unwrap();
        assert_eq!(read_utf32s, expected_utf32s);
    }

    check_writing(&utf32_strings, &utf32_strings);
    check_writing(&strings, &utf32_strings);
}

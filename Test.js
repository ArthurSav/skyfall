import React, { Component } from 'react'
import { Text, View, ToolbarAndroid, StyleSheet, Image, FlatList } from 'react-native'

export default class Test extends Component {

    static navigationOptions = { header: null }

    render() {
        return (
            <View style={{ marginTop: 24 }}>


            <Image style={{ width: 50, height: 50 }} source={{ uri: 'https://facebook.github.io/react-native/docs/assets/favicon.png' }}
            />




            <Image style={{ width: 50, height: 50 }} source={{ uri: 'https://facebook.github.io/react-native/docs/assets/favicon.png' }}
            />


            </View>
        )
    }
}

const styles = StyleSheet.create({
    toolbar: {
        backgroundColor: '#2196F3',
        height: 56
    },
});
